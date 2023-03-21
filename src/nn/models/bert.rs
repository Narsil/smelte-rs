use crate::cpu::f32::{matmul, matmul_t, softmax, Tensor as F32Tensor};

#[cfg(feature = "gpu")]
use crate::gpu::f32 as cuda_f32;

#[cfg(feature = "gpu")]
use crate::gpu::f32::Tensor as F32CudaTensor;

use crate::nn::layers::{Embedding, LayerNorm, Linear};
use crate::traits::{Tensor, TensorOps};
use crate::SmeltError;

macro_rules! debug {
    // `()` indicates that the macro takes no argument.
    ($str: expr, $tensor: expr) => {
        // The macro will expand into the contents of this block.
        // let data = $tensor.cpu_data().unwrap();
        // println!(
        //     "{} {:?}..{:?}",
        //     $str,
        //     &data[..3],
        //     &data[data.len() - 3..]
        // );
        // let n = $tensor.data().len();
        // println!(
        //     "{} {:?}..{:?}",
        //     $str,
        //     &$tensor.data()[..3],
        //     &$tensor.data()[n - 768 * 3..n - 768 * 3 + 3]
        // );
    };
}

/// TODO
pub struct BertContext<T: Tensor> {
    input_ids: Vec<usize>,
    type_ids: Vec<usize>,
    position_ids: Vec<usize>,
    hidden_states: T,
    // Required to compute position_ids before adding into hidden_states
    // - Used in the MLP to prevent cloning the skip connection
    // - Used in the attention for the output Linear layer
    hidden_states_copy: T,
    // Store the hidden_states after the attention (prevents a clone in the skip connection)
    hidden_states_attn_output: T,
    q_cache: T,
    // Store the k splitted_heads
    k_cache: T,
    // Store the k splitted_heads
    v_cache: T,
    // Store the qk result
    qk: T,
    qkv: T,
    // Intermediate states (H, 4H)
    intermediate_states: T,
    pool: T,
    pool_output: T,
    probs: T,
}

impl<T: Tensor> BertContext<T> {
    /// TODO
    pub fn probs(&self) -> &T {
        &self.probs
    }
}

fn split_heads(q: &F32Tensor, out_q: &mut F32Tensor) -> Result<(), SmeltError> {
    let num_heads = out_q.shape()[0];
    let sequence_length = out_q.shape()[1];
    let head_dim = out_q.shape()[2];
    let hidden_dim = head_dim * num_heads;

    (0..num_heads).for_each(|i| {
        (0..sequence_length).for_each(|j| {
            (0..head_dim).for_each(|k| {
                let index = j * hidden_dim + i * head_dim + k;
                let out_index = i * sequence_length * head_dim + j * head_dim + k;
                out_q.data_mut()[out_index] = q.data()[index];
            });
        });
    });
    Ok(())
}

#[inline]
fn unsplit_heads(src: &F32Tensor, dst: &mut F32Tensor) -> Result<(), SmeltError>{
    let num_heads = src.shape()[0];
    let sequence_length = src.shape()[1];
    let head_dim = src.shape()[2];
    let hidden_dim = head_dim * num_heads;
    (0..num_heads).for_each(|i| {
        (0..sequence_length).for_each(|j| {
            (0..head_dim).for_each(|k| {
                let in_index = i * sequence_length * head_dim + j * head_dim + k;
                let out_index = j * hidden_dim + i * head_dim + k;
                dst.data_mut()[out_index] = src.data()[in_index];
            });
        });
    });
    Ok(())
}

fn attention<'data, 'ctx>(
    q_weights: &Linear<F32Tensor<'data>>,
    k_weights: &Linear<F32Tensor<'data>>,
    v_weights: &Linear<F32Tensor<'data>>,
    ctx: &mut BertContext<F32Tensor<'ctx>>,
) -> Result<(), SmeltError>
where
    'data: 'ctx,
{
    q_weights.forward(&ctx.hidden_states, &mut ctx.hidden_states_copy)?;
    split_heads(&ctx.hidden_states_copy, &mut ctx.q_cache)?;

    debug!("Q head splitted", ctx.q_cache);

    k_weights.forward(&ctx.hidden_states, &mut ctx.hidden_states_copy)?;
    split_heads(&ctx.hidden_states_copy, &mut ctx.k_cache)?;

    debug!("K head splitted", ctx.k_cache);

    v_weights.forward(&ctx.hidden_states, &mut ctx.hidden_states_copy)?;
    split_heads(&ctx.hidden_states_copy, &mut ctx.v_cache)?;

    debug!("V head splitted", ctx.v_cache);

    matmul_t(&ctx.q_cache, &ctx.k_cache, &mut ctx.qk).unwrap();

    // let num_heads = ctx.q_cache.shape()[0];
    // let sequence_length = ctx.q_cache.shape()[1];
    let head_dim = ctx.q_cache.shape()[2];
    // let hidden_dim = head_dim * num_heads;
    let scale = (head_dim as f32).sqrt();
    ctx.qk.data_mut().iter_mut().for_each(|v| *v /= scale);

    softmax(&mut ctx.qk).unwrap();
    debug!("attention_probs", ctx.qk);
    matmul(&ctx.qk, &ctx.v_cache, &mut ctx.qkv)?;
    debug!("qkv", ctx.qkv);

    unsplit_heads(&ctx.qkv, &mut ctx.hidden_states_attn_output)?;

    debug!("qkv (reshaed)", ctx.hidden_states_attn_output);

    Ok(())
}

#[cfg(feature = "gpu")]
mod cuda {
    use super::*;
    use crate::gpu::f32::CudaError;
    use cudarc::driver::{DeviceSlice, LaunchAsync, LaunchConfig};

    const RESHAPE_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/bert_reshape.ptx"));

    pub(super) fn cuda_split_heads(src: &F32CudaTensor, dst: &mut F32CudaTensor) -> Result<(), SmeltError> {
        let dev = src.device();
        if src.device_id() != dst.device_id() {
            return Err(SmeltError::Cuda(CudaError::TensorOnDifferentDevice {
                got: src.device_id(),
                expected: dst.device_id(),
            }));
        }
        let module_name = "split_heads";
        if !dev.has_func(module_name, module_name) {
            dev.load_ptx(RESHAPE_PTX.into(), module_name, &[module_name])?;
        }

        let numel = dst.data().len();
        let num_heads = dst.shape()[0];
        let sequence_length = dst.shape()[1];
        let head_dim = dst.shape()[2];

        let fwd_fn = dev.get_func(module_name, module_name).unwrap();
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let params = (
            numel,
            src.data(),
            dst.data_mut(),
            num_heads,
            sequence_length,
            head_dim,
        );
        unsafe { fwd_fn.launch(cfg, params) }?;

        Ok(())
    }

    pub(super) fn cuda_unsplit_heads(src: &F32CudaTensor, dst: &mut F32CudaTensor) -> Result<(), SmeltError> {
        let dev = src.device();
        if src.device_id() != dst.device_id() {
            return Err(SmeltError::Cuda(CudaError::TensorOnDifferentDevice {
                got: src.device_id(),
                expected: dst.device_id(),
            }));
        }
        let module_name = "unsplit_heads";
        if !dev.has_func(module_name, module_name) {
            dev.load_ptx(RESHAPE_PTX.into(), module_name, &[module_name])?;
        }

        let numel = src.data().len();
        let num_heads = src.shape()[0];
        let sequence_length = src.shape()[1];
        let head_dim = src.shape()[2];

        let fwd_fn = dev.get_func(module_name, module_name).unwrap();
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let params = (
            numel,
            src.data(),
            dst.data_mut(),
            num_heads,
            sequence_length,
            head_dim,
        );
        unsafe { fwd_fn.launch(cfg, params) }?;

        Ok(())
    }
    fn cuda_attention(
        q_weights: &Linear<F32CudaTensor>,
        k_weights: &Linear<F32CudaTensor>,
        v_weights: &Linear<F32CudaTensor>,
        ctx: &mut BertContext<F32CudaTensor>,
    ) -> Result<(), SmeltError> {
        q_weights.forward(&ctx.hidden_states, &mut ctx.hidden_states_copy)?;
        cuda_split_heads(&ctx.hidden_states_copy, &mut ctx.q_cache)?;

        debug!("Q head splitted", ctx.q_cache);

        k_weights.forward(&ctx.hidden_states, &mut ctx.hidden_states_copy)?;
        cuda_split_heads(&ctx.hidden_states_copy, &mut ctx.k_cache)?;

        debug!("K head splitted", ctx.k_cache);

        v_weights.forward(&ctx.hidden_states, &mut ctx.hidden_states_copy)?;
        cuda_split_heads(&ctx.hidden_states_copy, &mut ctx.v_cache)?;

        debug!("V head splitted", ctx.v_cache);

        cuda_f32::matmul_t(&ctx.q_cache, &ctx.k_cache, &mut ctx.qk)?;

        let head_dim = ctx.q_cache.shape()[2];
        let scale = (head_dim as f32).sqrt();
        cuda_f32::mul_scalar(&mut ctx.qk, 1.0 / scale)?;

        cuda_f32::softmax(&mut ctx.qk)?;
        debug!("attention_probs", ctx.qk);
        cuda_f32::matmul(&ctx.qk, &ctx.v_cache, &mut ctx.qkv)?;
        debug!("qkv", ctx.qkv);

        cuda_unsplit_heads(&ctx.qkv, &mut ctx.hidden_states_attn_output)?;

        Ok(())
    }
    impl TensorAttention<F32CudaTensor> for F32CudaTensor {
        fn attention(
            query: &Linear<F32CudaTensor>,
            key: &Linear<F32CudaTensor>,
            value: &Linear<F32CudaTensor>,
            ctx: &mut BertContext<F32CudaTensor>,
        ) -> Result<(), SmeltError> {
            cuda_attention(query, key, value, ctx)?;
            Ok(())
        }
    }

    impl TensorDebug<F32CudaTensor> for F32CudaTensor {
        fn cpu_data(
            &self
        ) -> Result<Vec<f32>, SmeltError> {
            self.cpu_data()
        }
    }


    impl BertOps<F32CudaTensor> for F32CudaTensor {}
}

/// TODO
pub trait TensorAttention<T: Tensor> {
    /// TODO
    fn attention(
        query: &Linear<T>,
        key: &Linear<T>,
        value: &Linear<T>,
        ctx: &mut BertContext<T>,
    ) -> Result<(), SmeltError>;
}

/// TODO
pub trait TensorDebug<T: Tensor> {
    /// TODO
    fn cpu_data(
        &self
    ) -> Result<Vec<f32>, SmeltError>;
}

impl<'a> TensorAttention<F32Tensor<'a>> for F32Tensor<'a> {
    fn attention(
        query: &Linear<F32Tensor<'a>>,
        key: &Linear<F32Tensor<'a>>,
        value: &Linear<F32Tensor<'a>>,
        ctx: &mut BertContext<F32Tensor<'a>>,
    ) -> Result<(), SmeltError> {
        attention(query, key, value, ctx)?;
        Ok(())
    }
}

impl<'a> TensorDebug<F32Tensor<'a>> for F32Tensor<'a> {
    fn cpu_data(
        &self
    ) -> Result<Vec<f32>, SmeltError> {
        Ok(self.data().to_vec())
    }
}

/// TODO
pub trait BertOps<T: Tensor>: TensorOps<T> + TensorAttention<T> + TensorDebug<T> {}

impl<'a> BertOps<F32Tensor<'a>> for F32Tensor<'a> {}

/// TODO
#[derive(Clone)]
pub struct BertAttention<T: Tensor> {
    query: Linear<T>,
    key: Linear<T>,
    value: Linear<T>,
    output: Linear<T>,
    output_ln: LayerNorm<T>,
}

impl<T: Tensor + BertOps<T>> BertAttention<T> {
    /// TODO
    pub fn new(
        query: Linear<T>,
        key: Linear<T>,
        value: Linear<T>,
        output: Linear<T>,
        output_ln: LayerNorm<T>,
    ) -> Self {
        Self {
            query,
            key,
            value,
            output,
            output_ln,
        }
    }

    /// TODO
    pub fn forward(&self, ctx: &mut BertContext<T>) -> Result<(), SmeltError> {
        T::attention(&self.query, &self.key, &self.value, ctx)?;

        self.output
            .forward(&ctx.hidden_states_attn_output, &mut ctx.hidden_states_copy)?;
        T::add(&ctx.hidden_states_copy, &mut ctx.hidden_states)?;
        self.output_ln.forward(&mut ctx.hidden_states)?;
        Ok(())
    }
}

/// TODO
#[derive(Clone)]
pub struct Mlp<T: Tensor> {
    intermediate: Linear<T>,
    output: Linear<T>,
    output_ln: LayerNorm<T>,
}

impl<T: Tensor + BertOps<T>> Mlp<T> {
    /// TODO
    pub fn new(intermediate: Linear<T>, output: Linear<T>, output_ln: LayerNorm<T>) -> Self {
        Self {
            intermediate,
            output,
            output_ln,
        }
    }

    /// TODO
    pub fn forward(&self, ctx: &mut BertContext<T>) -> Result<(), SmeltError> {
        // println!("=====");
        debug!("Before MLP", ctx.hidden_states);
        self.intermediate
            .forward(&ctx.hidden_states, &mut ctx.intermediate_states)?;
        debug!("Intermediate ", ctx.intermediate_states);
        T::gelu(&mut ctx.intermediate_states)?;
        debug!("Intermediate (gelu)", ctx.intermediate_states);
        self.output
            .forward(&ctx.intermediate_states, &mut ctx.hidden_states_copy)?;
        debug!("output", ctx.hidden_states_copy);
        T::add(&ctx.hidden_states_copy, &mut ctx.hidden_states)?;
        debug!("output (skip)", ctx.hidden_states);
        self.output_ln.forward(&mut ctx.hidden_states)?;
        debug!("output ln", ctx.hidden_states);
        Ok(())
    }
}

/// TODO
#[derive(Clone)]
pub struct BertLayer<T: Tensor> {
    attention: BertAttention<T>,
    mlp: Mlp<T>,
}

impl<T: Tensor + BertOps<T>> BertLayer<T> {
    /// TODO
    pub fn new(attention: BertAttention<T>, mlp: Mlp<T>) -> Self {
        Self { attention, mlp }
    }

    /// TODO
    pub fn forward(&self, ctx: &mut BertContext<T>) -> Result<(), SmeltError> {
        debug!("Before attention", ctx.hidden_states);
        self.attention.forward(ctx)?;
        debug!("After attention", ctx.hidden_states);
        self.mlp.forward(ctx)?;
        debug!("After mlp", ctx.hidden_states);
        // println!("---------");
        Ok(())
    }
}

/// TODO
#[derive(Clone)]
pub struct BertEncoder<T: Tensor> {
    layers: Vec<BertLayer<T>>,
}

impl<T: Tensor + BertOps<T>> BertEncoder<T> {
    /// TODO
    pub fn new(layers: Vec<BertLayer<T>>) -> Self {
        Self { layers }
    }

    /// TODO
    pub fn forward(&self, ctx: &mut BertContext<T>) -> Result<(), SmeltError> {
        for layer in &self.layers {
            layer.forward(ctx)?;
        }
        Ok(())
    }
}

/// TODO
#[derive(Clone)]
pub struct BertEmbeddings<T: Tensor> {
    input_embeddings: Embedding<T>,
    position_embeddings: Embedding<T>,
    type_embeddings: Embedding<T>,
    layer_norm: LayerNorm<T>,
}

impl<T: Tensor + BertOps<T>> BertEmbeddings<T> {
    /// TODO
    pub fn new(
        input_embeddings: Embedding<T>,
        position_embeddings: Embedding<T>,
        type_embeddings: Embedding<T>,
        layer_norm: LayerNorm<T>,
    ) -> Self {
        Self {
            input_embeddings,
            position_embeddings,
            type_embeddings,
            layer_norm,
        }
    }

    /// TODO
    pub fn forward(&self, ctx: &mut BertContext<T>) -> Result<(), SmeltError> {
        let input_ids = &ctx.input_ids;
        let position_ids = &ctx.position_ids;
        let type_ids = &ctx.type_ids;

        if input_ids.len() != position_ids.len() {
            return Err(SmeltError::InvalidLength {
                expected: input_ids.len(),
                got: position_ids.len(),
            });
        }
        if input_ids.len() != type_ids.len() {
            return Err(SmeltError::InvalidLength {
                expected: input_ids.len(),
                got: type_ids.len(),
            });
        }

        self.input_embeddings
            .forward(input_ids, &mut ctx.hidden_states)?;

        debug!("input embeddings", ctx.hidden_states);

        self.type_embeddings
            .forward(type_ids, &mut ctx.hidden_states_copy)?;
        debug!("type embeddings", ctx.hidden_states_copy);
        T::add(&ctx.hidden_states_copy, &mut ctx.hidden_states)?;
        debug!("After add type embeddings", ctx.hidden_states);

        self.position_embeddings
            .forward(position_ids, &mut ctx.hidden_states_copy)?;
        debug!("position embeddings", ctx.hidden_states_copy);
        T::add(&ctx.hidden_states_copy, &mut ctx.hidden_states)?;
        debug!("After add position embeddings", ctx.hidden_states);

        self.layer_norm.forward(&mut ctx.hidden_states)?;

        debug!("After embeddings", ctx.hidden_states);
        Ok(())
    }
}

/// TODO
#[derive(Clone)]
pub struct Bert<T: Tensor + BertOps<T>> {
    embeddings: BertEmbeddings<T>,
    encoder: BertEncoder<T>,
}

impl<T: Tensor + BertOps<T>> Bert<T> {
    /// TODO
    pub fn new(embeddings: BertEmbeddings<T>, encoder: BertEncoder<T>) -> Self {
        Self {
            embeddings,
            encoder,
        }
    }
    /// TODO
    pub fn forward(&self, ctx: &mut BertContext<T>) -> Result<(), SmeltError> {
        self.embeddings.forward(ctx)?;
        self.encoder.forward(ctx)
    }
}

/// TODO
#[derive(Clone)]
pub struct BertPooler<T: Tensor> {
    pooler: Linear<T>,
}

impl<T: Tensor + BertOps<T>> BertPooler<T> {
    /// TODO
    pub fn new(pooler: Linear<T>) -> Self {
        Self { pooler }
    }

    /// TODO
    pub fn forward(&self, ctx: &mut BertContext<T>) -> Result<(), SmeltError> {
        T::select(&[0], &ctx.hidden_states, &mut ctx.pool)?;
        self.pooler.forward(&ctx.pool, &mut ctx.pool_output)?;
        T::tanh(&mut ctx.pool_output)?;
        Ok(())
    }
}

/// TODO
#[derive(Clone)]
pub struct BertClassifier<T: Tensor + BertOps<T>> {
    bert: Bert<T>,
    pooler: BertPooler<T>,
    classifier: Linear<T>,
    num_heads: usize,
}

impl<T: Tensor + BertOps<T> + TensorAttention<T>> BertClassifier<T> {
    /// TODO
    pub fn new(bert: Bert<T>, pooler: BertPooler<T>, classifier: Linear<T>) -> Self {
        Self {
            bert,
            pooler,
            classifier,
            num_heads: 0,
        }
    }

    /// TODO
    pub fn set_num_heads(&mut self, num_heads: usize) {
        self.num_heads = num_heads
    }

    /// TODO
    pub fn forward(&self, ctx: &mut BertContext<T>) -> Result<(), SmeltError> {
        self.bert.forward(ctx)?;
        self.pooler.forward(ctx)?;
        self.classifier.forward(&ctx.pool_output, &mut ctx.probs)?;
        T::softmax(&mut ctx.probs)?;
        Ok(())
    }

    /// TODO
    pub fn new_context(
        &self,
        input_ids: Vec<usize>,
        position_ids: Vec<usize>,
        type_ids: Vec<usize>,
        num_heads: usize,
    ) -> BertContext<T> {
        let hidden_dim = self.bert.embeddings.input_embeddings.weight().shape()[1];
        let intermediate_dim = self.bert.encoder.layers[0]
            .mlp
            .intermediate
            .weight()
            .shape()[0];
        let num_classes = self.classifier.weight().shape()[0];
        let head_dim = hidden_dim / num_heads;
        let sequence_length = input_ids.len();

        let hidden_states = T::zeros(vec![sequence_length, hidden_dim]);
        let hidden_states_copy = T::zeros(vec![sequence_length, hidden_dim]);
        let hidden_states_attn_output = T::zeros(vec![sequence_length, hidden_dim]);
        let intermediate_states = T::zeros(vec![sequence_length, intermediate_dim]);
        let q_cache = T::zeros(vec![num_heads, sequence_length, head_dim]);
        let k_cache = T::zeros(vec![num_heads, sequence_length, head_dim]);
        let v_cache = T::zeros(vec![num_heads, sequence_length, head_dim]);
        let qk = T::zeros(vec![num_heads, sequence_length, sequence_length]);
        let qkv = T::zeros(vec![num_heads, sequence_length, head_dim]);
        let pool = T::zeros(vec![1, hidden_dim]);
        let pool_output = T::zeros(vec![1, hidden_dim]);
        let probs = T::zeros(vec![1, num_classes]);
        BertContext {
            input_ids,
            position_ids,
            type_ids,
            hidden_states,
            hidden_states_copy,
            hidden_states_attn_output,
            intermediate_states,
            q_cache,
            k_cache,
            v_cache,
            qk,
            qkv,
            pool,
            pool_output,
            probs,
        }
    }

    /// TODO
    pub fn run(
        &self,
        input_ids: Vec<usize>,
        position_ids: Vec<usize>,
        type_ids: Vec<usize>,
    ) -> Result<T, SmeltError> {
        let mut context = self.new_context(input_ids, position_ids, type_ids, self.num_heads);
        self.forward(&mut context)?;
        Ok(context.probs)
    }
}


#[cfg(test)]
mod tests{
    use super::*;

    #[test]
    fn test_split_heads(){
        let tensor = F32Tensor::new(vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0], vec![2, 4]).unwrap();
        let mut out = F32Tensor::zeros(vec![2, 2, 2]);

        split_heads(&tensor, &mut out).unwrap();
        assert_eq!(out.data(), [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0]);

    }

    #[test]
    fn test_unsplit_heads(){
        let tensor = F32Tensor::new(vec![1.0, 3.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0], vec![2, 2, 2]).unwrap();
        let mut out = F32Tensor::zeros(vec![2, 4]);

        unsplit_heads(&tensor, &mut out).unwrap();
        assert_eq!(out.data(),[1.0, 3.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0] );

    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_cuda_split_heads(){
        let tensor = F32CudaTensor::from_cpu(&[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0], vec![2, 4], 0).unwrap();
        let mut out = F32CudaTensor::zeros(vec![2, 2, 2], 0).unwrap();

        cuda::cuda_split_heads(&tensor, &mut out).unwrap();
        assert_eq!(out.cpu_data().unwrap(), vec![1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0]);

    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_cuda_unsplit_heads(){
        let tensor = F32CudaTensor::from_cpu(&[1.0, 3.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0], vec![2, 2, 2], 0).unwrap();
        let mut out = F32CudaTensor::zeros(vec![2, 4], 0).unwrap();

        cuda::cuda_unsplit_heads(&tensor, &mut out).unwrap();
        assert_eq!(out.cpu_data().unwrap(), vec![1.0, 3.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]);

    }


}
