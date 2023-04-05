// use crate::cpu::f32::{Tensor as F32Tensor};
#[cfg(feature = "cpu")]
use crate::cpu::f32::Tensor as F32Tensor;

#[cfg(feature = "cuda")]
use crate::gpu::f32 as cuda_f32;
#[cfg(feature = "cuda")]
use cudarc::driver::{profiler_start, profiler_stop};

#[cfg(feature = "cuda")]
use crate::gpu::f32::Tensor as F32CudaTensor;

use crate::nn::layers::{Embedding, LayerNorm, LinearT, UnbiasedLinear};
use crate::traits::{Device, Tensor, TensorOps};
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

/// A special structure handy for Past Key values for text-generation
pub struct PastKeyValue<T: Tensor> {
    /// The cached key tensor. Shape is `NUM_HEADS, PAST_SEQUENCE_LENGTH, HEAD_DIM`.
    pub key: T,
    /// The cached value tensor. Shape is `NUM_HEADS, PAST_SEQUENCE_LENGTH, HEAD_DIM`.
    pub value: T,
}

impl<T: Tensor> PastKeyValue<T> {
    /// TODO
    pub fn new<D: Device<Tensor = T>>(
        num_heads: usize,
        past_sequence_length: usize,
        head_dim: usize,
        device: &D,
    ) -> Result<Self, SmeltError> {
        let key = device.zeros(vec![num_heads, past_sequence_length, head_dim])?;
        let value = device.zeros(vec![num_heads, past_sequence_length, head_dim])?;
        Ok(Self { key, value })
    }
}

/// TODO
pub type PastKeyValues<T> = Vec<PastKeyValue<T>>;

/// TODO
pub struct Gpt2Context<T: Tensor> {
    input_ids: Vec<usize>,
    position_ids: Vec<usize>,
    hidden_states: T,
    // Required to compute position_ids before adding into hidden_states
    // - Used in the MLP to prevent cloning the skip connection
    // - Used in the attention for the output LinearT layer
    hidden_states_copy: T,
    qkv: T,
    q: T,
    k: T,
    v: T,
    qk: T,
    qkv_out: T,
    past_key_values: PastKeyValues<T>,
    intermediate_states: T,
    probs: T,
}

impl<T: Tensor> Gpt2Context<T> {
    /// TODO
    pub fn new_tokens(&self) -> Result<Vec<usize>, SmeltError> {
        Ok(vec![])
    }

    /// TODO
    pub fn new<D: Device<Tensor = T>>(
        input_ids: Vec<usize>,
        num_heads: usize,
        vocab_size: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
        num_layers: usize,
        device: &D,
    ) -> Result<Self, SmeltError> {
        let position_ids: Vec<_> = (0..input_ids.len()).collect();
        let head_dim = hidden_dim / num_heads;
        let sequence_length = input_ids.len();
        let past_length = 0;
        let past_key_values: Result<PastKeyValues<T>, _> = (0..num_layers)
            .map(|_| -> Result<PastKeyValue<T>, _> {
                PastKeyValue::new(num_heads, past_length, head_dim, device)
            })
            .collect();
        let past_key_values = past_key_values?;

        let hidden_states = device.zeros(vec![sequence_length, hidden_dim])?;
        let hidden_states_copy = device.zeros(vec![sequence_length, hidden_dim])?;
        let intermediate_states = device.zeros(vec![sequence_length, intermediate_dim])?;
        let qkv = device.zeros(vec![sequence_length, hidden_dim * 3])?;
        let q = device.zeros(vec![num_heads, sequence_length, head_dim])?;
        let k = device.zeros(vec![num_heads, sequence_length + past_length, head_dim])?;
        let v = device.zeros(vec![num_heads, sequence_length + past_length, head_dim])?;
        let qk = device.zeros(vec![
            num_heads,
            sequence_length,
            sequence_length + past_length,
        ])?;
        let qkv_out = device.zeros(vec![num_heads, sequence_length, head_dim])?;
        let probs = device.zeros(vec![sequence_length, vocab_size])?;
        Ok(Gpt2Context {
            input_ids,
            position_ids,
            hidden_states,
            hidden_states_copy,
            q,
            k,
            v,
            qk,
            qkv_out,
            intermediate_states,
            past_key_values,
            qkv,
            probs,
        })
    }
    /// TODO
    pub fn next(&mut self) -> Result<(), SmeltError> {
        let past_length = self.past_key_values[0].key.shape()[1];
        let position_ids: Vec<_> = vec![past_length + 1];
        let hidden_dim = self.hidden_states.shape()[1];
        let device = self.q.device();
        let num_heads = self.q.shape()[0];
        let head_dim = self.q.shape()[2];
        let intermediate_dim = self.intermediate_states.shape()[1];
        let vocab_size = self.probs.shape()[1];
        let sequence_length = 1;
        let num_layers = self.past_key_values.len();
        let past_key_values: Result<PastKeyValues<T>, _> = (0..num_layers)
            .map(|_| -> Result<PastKeyValue<T>, _> {
                PastKeyValue::new(num_heads, past_length, head_dim, device)
            })
            .collect();
        let past_key_values = past_key_values?;

        let hidden_states = device.zeros(vec![sequence_length, hidden_dim])?;
        let hidden_states_copy = device.zeros(vec![sequence_length, hidden_dim])?;
        let intermediate_states = device.zeros(vec![sequence_length, intermediate_dim])?;
        let qkv = device.zeros(vec![sequence_length, hidden_dim * 3])?;
        let q = device.zeros(vec![num_heads, sequence_length, head_dim])?;
        let k = device.zeros(vec![num_heads, sequence_length + past_length, head_dim])?;
        let v = device.zeros(vec![num_heads, sequence_length + past_length, head_dim])?;
        let qk = device.zeros(vec![
            num_heads,
            sequence_length,
            sequence_length + past_length,
        ])?;
        let qkv_out = device.zeros(vec![num_heads, sequence_length, head_dim])?;
        let probs = device.zeros(vec![sequence_length, vocab_size])?;
        // TODO
        self.input_ids = vec![0];
        self.position_ids = position_ids;
        self.hidden_states = hidden_states;
        self.hidden_states_copy = hidden_states_copy;
        self.q = q;
        self.k = k;
        self.v = v;
        self.qkv = qkv;
        self.qkv_out = qkv_out;
        self.intermediate_states = intermediate_states;
        self.qk = qk;
        self.probs = probs;
        Ok(())
    }

    /// TODO
    pub fn probs(&self) -> &T {
        &self.probs
    }
}

#[cfg(feature = "cpu")]
mod cpu {
    use super::*;
    use crate::cpu::f32::{causal_softmax, matmul, matmul_t};

    fn split_qkv(ctx: &mut Gpt2Context<F32Tensor>, i: usize) -> Result<(), SmeltError> {
        let sequence_length = ctx.qkv.shape()[0];
        let past_sequence_length = ctx.past_key_values[i].key.shape()[1];
        let hidden_dim3 = ctx.qkv.shape()[1];
        assert_eq!(hidden_dim3 % 3, 0);
        let hidden_dim = hidden_dim3 / 3;
        let num_heads = ctx.past_key_values[i].key.shape()[0];
        assert_eq!(hidden_dim % num_heads, 0);
        let head_dim = hidden_dim / num_heads;
        let query_data = ctx.q.data_mut();
        (0..num_heads).for_each(|i| {
            (0..sequence_length).for_each(|j| {
                (0..head_dim).for_each(|k| {
                    let index = j * hidden_dim * 3 + i * head_dim + k;
                    let out_index = i * sequence_length * head_dim + j * head_dim + k;
                    let value = ctx.qkv.data()[index];
                    query_data[out_index] = value;
                });
            });
        });

        let key_data = ctx.k.data_mut();
        let value_data = ctx.v.data_mut();
        (0..num_heads).for_each(|i| {
            (0..past_sequence_length + sequence_length).for_each(|j| {
                (0..head_dim).for_each(|k| {
                    let in_index =
                        i * (past_sequence_length + sequence_length) * head_dim + j * head_dim + k;
                    if j < past_sequence_length {
                        let index = i * past_sequence_length * head_dim + j * head_dim + k;
                        let k_value = ctx.past_key_values[i].key.data()[index];
                        let v_value = ctx.past_key_values[i].value.data()[index];
                        key_data[in_index] = k_value;
                        value_data[in_index] = v_value;
                    } else {
                        let sj = j - past_sequence_length;
                        let k_index = sj * hidden_dim * 3 + i * head_dim + hidden_dim + k;
                        let v_index = sj * hidden_dim * 3 + i * head_dim + hidden_dim * 2 + k;
                        let k_value = ctx.qkv.data()[k_index];
                        let v_value = ctx.qkv.data()[v_index];
                        key_data[in_index] = k_value;
                        value_data[in_index] = v_value;
                    }
                });
            });
        });

        ctx.past_key_values[i].key = ctx.k.clone();
        ctx.past_key_values[i].value = ctx.v.clone();
        Ok(())
    }

    fn attention(
        qkv_weights: &LinearT<F32Tensor>,
        ctx: &mut Gpt2Context<F32Tensor>,
        i: usize,
    ) -> Result<(), SmeltError> {
        let sequence_length = ctx.qkv.shape()[0];
        let past_sequence_length = ctx.past_key_values[i].key.shape()[1];
        let hidden_dim3 = ctx.qkv.shape()[1];
        assert_eq!(hidden_dim3 % 3, 0);
        let hidden_dim = hidden_dim3 / 3;
        let num_heads = ctx.qk.shape()[0];
        assert_eq!(hidden_dim % num_heads, 0);
        let head_dim = hidden_dim / num_heads;
        qkv_weights
            .forward(&ctx.hidden_states, &mut ctx.qkv)
            .unwrap();

        assert_eq!(
            ctx.qk.shape(),
            vec![
                num_heads,
                sequence_length,
                (past_sequence_length + sequence_length)
            ]
        );
        assert_eq!(
            ctx.past_key_values[i].key.shape(),
            vec![num_heads, past_sequence_length, head_dim]
        );
        assert_eq!(
            ctx.past_key_values[i].value.shape(),
            vec![num_heads, past_sequence_length, head_dim]
        );

        split_qkv(ctx, i)?;
        matmul_t(&ctx.q, &ctx.k, &mut ctx.qk).unwrap();
        let head_dim = hidden_dim / num_heads;
        let scale = (head_dim as f32).sqrt();
        ctx.qk.data_mut().iter_mut().for_each(|v| *v /= scale);
        causal_softmax(&mut ctx.qk, past_sequence_length).unwrap();
        matmul(&ctx.qk, &ctx.v, &mut ctx.qkv_out).unwrap();

        let new_out = ctx.hidden_states.data_mut();
        (0..num_heads).for_each(|i| {
            (0..sequence_length).for_each(|j| {
                (0..head_dim).for_each(|k| {
                    let in_index = i * sequence_length * head_dim + j * head_dim + k;
                    let out_index = j * hidden_dim + i * head_dim + k;
                    new_out[out_index] = ctx.qkv_out.data()[in_index];
                });
            });
        });
        Ok(())
    }

    impl TensorAttention<F32Tensor> for F32Tensor {
        fn attention(
            qkv: &LinearT<F32Tensor>,
            ctx: &mut Gpt2Context<F32Tensor>,
            i: usize,
        ) -> Result<(), SmeltError> {
            attention(qkv, ctx, i)?;
            Ok(())
        }
    }

    impl TensorDebug<F32Tensor> for F32Tensor {
        fn cpu_data(&self) -> Result<Vec<f32>, SmeltError> {
            Ok(self.data().to_vec())
        }
    }

    impl Gpt2Ops<F32Tensor> for F32Tensor {}
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use crate::gpu::f32::{causal_softmax, matmul, matmul_t};
    use cudarc::driver::{DeviceSlice, LaunchAsync, LaunchConfig};

    const RESHAPE_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/gpt2_reshape.ptx"));

    fn split_qkv(ctx: &mut Gpt2Context<F32CudaTensor>, i: usize) -> Result<(), SmeltError> {
        let dev = ctx.qkv.cuda();
        let module_name = "attention_reshape";
        if !dev.has_func(module_name, module_name) {
            dev.load_ptx(RESHAPE_PTX.into(), module_name, &[module_name])?;
        }
        let numel = ctx.qkv.data().len();
        let num_heads = ctx.q.shape()[0];
        let sequence_length = ctx.q.shape()[1];
        let past_length = ctx.past_key_values[i].key.shape()[1];
        let head_dim = ctx.q.shape()[2];
        let fwd_fn = dev.get_func(module_name, module_name).unwrap();
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let params = (
            numel,
            num_heads,
            head_dim,
            sequence_length,
            past_length,
            ctx.qkv.data(),
            ctx.past_key_values[i].key.data(),
            ctx.past_key_values[i].value.data(),
            ctx.q.data_mut(),
            ctx.k.data_mut(),
            ctx.v.data_mut(),
        );
        unsafe { fwd_fn.launch(cfg, params) }?;

        Ok(())
    }

    fn unsplit(ctx: &mut Gpt2Context<F32CudaTensor>) -> Result<(), SmeltError> {
        let module_name = "attention_unsplit";
        let dev = ctx.qkv_out.cuda();
        if !dev.has_func(module_name, module_name) {
            dev.load_ptx(RESHAPE_PTX.into(), module_name, &[module_name])?;
        }
        let numel = ctx.qkv_out.data().len();
        let num_heads = ctx.q.shape()[0];
        let sequence_length = ctx.q.shape()[1];
        let head_dim = ctx.q.shape()[2];
        let fwd_fn = dev.get_func(module_name, module_name).unwrap();
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let params = (
            numel,
            num_heads,
            head_dim,
            sequence_length,
            ctx.qkv_out.data(),
            ctx.hidden_states.data_mut(),
        );
        unsafe { fwd_fn.launch(cfg, params) }?;

        Ok(())
    }

    fn cuda_attention(
        qkv: &LinearT<F32CudaTensor>,
        ctx: &mut Gpt2Context<F32CudaTensor>,
        i: usize,
    ) -> Result<(), SmeltError> {
        qkv.forward(&ctx.hidden_states, &mut ctx.qkv)?;

        split_qkv(ctx, i)?;
        matmul_t(&ctx.q, &ctx.k, &mut ctx.qk).unwrap();
        let hidden_dim = ctx.qkv.shape()[1];
        let num_heads = ctx.q.shape()[0];
        let past_sequence_length = ctx.past_key_values[i].key.shape()[1];
        let head_dim = hidden_dim / num_heads;
        let scale = (head_dim as f32).sqrt();
        cuda_f32::mul_scalar(&mut ctx.qk, 1.0 / scale)?;
        causal_softmax(&mut ctx.qk, past_sequence_length).unwrap();
        matmul(&ctx.qk, &ctx.v, &mut ctx.qkv_out).unwrap();
        unsplit(ctx)?;
        Ok(())

        // Ok(())
    }
    impl TensorAttention<F32CudaTensor> for F32CudaTensor {
        fn attention(
            qkv: &LinearT<F32CudaTensor>,
            ctx: &mut Gpt2Context<F32CudaTensor>,
            i: usize,
        ) -> Result<(), SmeltError> {
            cuda_attention(qkv, ctx, i)?;
            Ok(())
        }
    }

    impl TensorDebug<F32CudaTensor> for F32CudaTensor {
        fn cpu_data(&self) -> Result<Vec<f32>, SmeltError> {
            self.cpu_data()
        }
    }

    impl Gpt2Ops<F32CudaTensor> for F32CudaTensor {}
}

/// TODO
pub trait TensorAttention<T: Tensor> {
    /// TODO
    fn attention(qkv: &LinearT<T>, ctx: &mut Gpt2Context<T>, i: usize) -> Result<(), SmeltError>;
}

/// TODO
pub trait TensorDebug<T: Tensor> {
    /// TODO
    fn cpu_data(&self) -> Result<Vec<f32>, SmeltError>;
}

/// TODO
pub trait Gpt2Ops<T: Tensor>: TensorOps<T> + TensorAttention<T> + TensorDebug<T> {}

/// TODO
#[derive(Clone)]
pub struct Gpt2Attention<T: Tensor> {
    qkv: LinearT<T>,
    output: LinearT<T>,
    i: usize,
}

impl<T: Tensor + Gpt2Ops<T>> Gpt2Attention<T> {
    /// TODO
    pub fn new(qkv: LinearT<T>, output: LinearT<T>, i: usize) -> Self {
        Self { qkv, output, i }
    }

    /// TODO
    pub fn forward(&self, ctx: &mut Gpt2Context<T>) -> Result<(), SmeltError> {
        T::attention(&self.qkv, ctx, self.i)?;

        self.output
            .forward(&ctx.hidden_states, &mut ctx.hidden_states_copy)?;
        T::add(&ctx.hidden_states_copy, &mut ctx.hidden_states)?;
        Ok(())
    }
}

/// TODO
#[derive(Clone)]
pub struct Mlp<T: Tensor> {
    c_fc: LinearT<T>,
    c_proj: LinearT<T>,
}

impl<T: Tensor + Gpt2Ops<T>> Mlp<T> {
    /// TODO
    pub fn new(c_fc: LinearT<T>, c_proj: LinearT<T>) -> Self {
        Self { c_fc, c_proj }
    }

    /// TODO
    pub fn forward(&self, ctx: &mut Gpt2Context<T>) -> Result<(), SmeltError> {
        // println!("=====");
        debug!("Before MLP", ctx.hidden_states);
        self.c_fc
            .forward(&ctx.hidden_states, &mut ctx.intermediate_states)
            .unwrap();
        debug!("Intermediate ", ctx.intermediate_states);
        T::gelu(&mut ctx.intermediate_states)?;
        debug!("Intermediate (gelu)", ctx.intermediate_states);
        self.c_proj
            .forward(&ctx.intermediate_states, &mut ctx.hidden_states)?;
        debug!("output ln", ctx.hidden_states);
        Ok(())
    }
}

/// TODO
#[derive(Clone)]
pub struct Gpt2Layer<T: Tensor> {
    attention: Gpt2Attention<T>,
    mlp: Mlp<T>,
    ln_1: LayerNorm<T>,
    ln_2: LayerNorm<T>,
}

impl<T: Tensor + Gpt2Ops<T>> Gpt2Layer<T> {
    /// TODO
    pub fn new(
        attention: Gpt2Attention<T>,
        mlp: Mlp<T>,
        ln_1: LayerNorm<T>,
        ln_2: LayerNorm<T>,
    ) -> Self {
        Self {
            attention,
            mlp,
            ln_1,
            ln_2,
        }
    }

    /// TODO
    pub fn forward(&self, ctx: &mut Gpt2Context<T>) -> Result<(), SmeltError> {
        T::copy(&ctx.hidden_states, &mut ctx.hidden_states_copy)?;
        self.ln_1.forward(&mut ctx.hidden_states)?;
        self.attention.forward(ctx)?;
        T::add(&ctx.hidden_states_copy, &mut ctx.hidden_states)?;
        T::copy(&ctx.hidden_states, &mut ctx.hidden_states_copy)?;
        self.ln_2.forward(&mut ctx.hidden_states)?;
        self.mlp.forward(ctx)?;
        T::add(&ctx.hidden_states_copy, &mut ctx.hidden_states)?;
        Ok(())
    }
}

/// TODO
#[derive(Clone)]
pub struct Gpt2Model<T: Tensor> {
    layers: Vec<Gpt2Layer<T>>,
}

impl<T: Tensor + Gpt2Ops<T>> Gpt2Model<T> {
    /// TODO
    pub fn new(layers: Vec<Gpt2Layer<T>>) -> Self {
        Self { layers }
    }

    /// TODO
    pub fn forward(&self, ctx: &mut Gpt2Context<T>) -> Result<(), SmeltError> {
        for layer in &self.layers {
            layer.forward(ctx)?;
        }
        Ok(())
    }
}

/// TODO
#[derive(Clone)]
pub struct Gpt2<T: Tensor + Gpt2Ops<T>> {
    wte: Embedding<T>,
    wpe: Embedding<T>,
    h: Gpt2Model<T>,
    ln_f: LayerNorm<T>,
    lm_head: UnbiasedLinear<T>,
    num_heads: usize,
}

impl<T: Tensor + Gpt2Ops<T>> Gpt2<T> {
    /// TODO
    pub fn new(
        wte: Embedding<T>,
        wpe: Embedding<T>,
        h: Gpt2Model<T>,
        ln_f: LayerNorm<T>,
        lm_head: UnbiasedLinear<T>,
        num_heads: usize,
    ) -> Self {
        Self {
            h,
            ln_f,
            wte,
            wpe,
            lm_head,
            num_heads,
        }
    }

    /// TODO
    pub fn set_num_heads(&mut self, num_heads: usize) {
        self.num_heads = num_heads;
    }

    /// TODO
    pub fn forward(&self, ctx: &mut Gpt2Context<T>) -> Result<(), SmeltError> {
        let input_ids = &ctx.input_ids;
        let position_ids = &ctx.position_ids;
        println!("Position ids {:?}", input_ids);
        if input_ids.len() != position_ids.len() {
            return Err(SmeltError::InvalidLength {
                expected: input_ids.len(),
                got: position_ids.len(),
            });
        }
        self.wte.forward(input_ids, &mut ctx.hidden_states)?;

        debug!("input embeddings", ctx.hidden_states);
        self.wpe
            .forward(position_ids, &mut ctx.hidden_states_copy)?;
        debug!("position embeddings", ctx.hidden_states_copy);
        T::add(&ctx.hidden_states_copy, &mut ctx.hidden_states)?;

        self.h.forward(ctx)?;
        self.ln_f.forward(&mut ctx.hidden_states)?;
        self.lm_head.forward(&ctx.hidden_states, &mut ctx.probs)?;
        Ok(())
    }

    /// TODO
    pub fn new_context(
        &self,
        input_ids: Vec<usize>,
        num_heads: usize,
    ) -> Result<Gpt2Context<T>, SmeltError> {
        let vocab_size = self.wte.weight().shape()[0];
        let hidden_dim = self.wte.weight().shape()[1];
        let intermediate_dim = self.h.layers[0].mlp.c_fc.weight().shape()[1];
        let num_layers = self.h.layers.len();
        let device = self.wte.weight().device();
        Gpt2Context::new(
            input_ids,
            num_heads,
            vocab_size,
            hidden_dim,
            intermediate_dim,
            num_layers,
            device,
        )
    }

    /// TODO
    pub fn run(&self, input_ids: Vec<usize>, new_tokens: usize) -> Result<Vec<usize>, SmeltError> {
        profiler_start()?;
        let mut context = self.new_context(input_ids, self.num_heads)?;
        for _ in 0..new_tokens {
            let start = std::time::Instant::now();
            self.forward(&mut context)?;
            context.next();
            println!("Took {:?}", start.elapsed());
        }
        let tokens = context.new_tokens();
        profiler_stop()?;
        tokens
    }
}
