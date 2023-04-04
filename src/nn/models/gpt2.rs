// use crate::cpu::f32::{Tensor as F32Tensor};
#[cfg(feature = "cpu")]
use crate::cpu::f32::Tensor as F32Tensor;

#[cfg(feature = "cuda")]
use crate::gpu::f32 as cuda_f32;

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
    past_key_values: PastKeyValues<T>,
    // Store the hidden_states after the attention (prevents a clone in the skip connection)
    hidden_states_attn_output: T,
    qkv_cache: T,
    intermediate_states: T,
    probs: T,
}

impl<T: Tensor> Gpt2Context<T> {
    /// TODO
    pub fn probs(&self) -> &T {
        &self.probs
    }
}

#[cfg(feature = "cpu")]
mod cpu {
    use super::*;

    fn attention(
        qkv_weights: &LinearT<F32Tensor>,
        ctx: &mut Gpt2Context<F32Tensor>,
    ) -> Result<(), SmeltError> {
        println!("{:?}", qkv_weights.weight().shape());
        println!("{:?}", ctx.qkv_cache.shape());
        println!("{:?}", ctx.hidden_states.shape());
        qkv_weights
            .forward(&ctx.hidden_states, &mut ctx.qkv_cache)
            .unwrap();
        assert_eq!(ctx.past_key_values.len(), 1);
        todo!();
        // Ok(())
    }

    impl TensorAttention<F32Tensor> for F32Tensor {
        fn attention(
            qkv: &LinearT<F32Tensor>,
            ctx: &mut Gpt2Context<F32Tensor>,
        ) -> Result<(), SmeltError> {
            attention(qkv, ctx)?;
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
    use crate::gpu::f32::CudaError;
    use cudarc::driver::{DeviceSlice, LaunchAsync, LaunchConfig};

    const RESHAPE_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/gpt2_reshape.ptx"));

    fn cuda_attention(
        qkv: &LinearT<F32CudaTensor>,
        ctx: &mut Gpt2Context<F32CudaTensor>,
    ) -> Result<(), SmeltError> {
        todo!("cuda gpt2");

        // Ok(())
    }
    impl TensorAttention<F32CudaTensor> for F32CudaTensor {
        fn attention(
            qkv: &LinearT<F32CudaTensor>,
            ctx: &mut Gpt2Context<F32CudaTensor>,
        ) -> Result<(), SmeltError> {
            cuda_attention(qkv, ctx)?;
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
    fn attention(qkv: &LinearT<T>, ctx: &mut Gpt2Context<T>) -> Result<(), SmeltError>;
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
}

impl<T: Tensor + Gpt2Ops<T>> Gpt2Attention<T> {
    /// TODO
    pub fn new(qkv: LinearT<T>, output: LinearT<T>) -> Self {
        Self { qkv, output }
    }

    /// TODO
    pub fn forward(&self, ctx: &mut Gpt2Context<T>) -> Result<(), SmeltError> {
        T::attention(&self.qkv, ctx)?;

        self.output
            .forward(&ctx.hidden_states_attn_output, &mut ctx.hidden_states_copy)?;
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
            .forward(&ctx.hidden_states, &mut ctx.intermediate_states)?;
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
        let position_ids: Vec<_> = (0..input_ids.len()).collect();
        let vocab_size = self.wpe.weight().shape()[0];
        let hidden_dim = self.wpe.weight().shape()[1];
        let intermediate_dim = self.h.layers[0].mlp.c_fc.weight().shape()[0];

        let head_dim = hidden_dim / num_heads;
        let sequence_length = input_ids.len();

        let device = self.wpe.weight().device();
        let past_key_values: Result<PastKeyValues<T>, _> = (0..self.h.layers.len())
            .map(|_| -> Result<PastKeyValue<T>, _> {
                PastKeyValue::new(num_heads, 0, head_dim, device)
            })
            .collect();
        let past_key_values = past_key_values?;

        let hidden_states = device.zeros(vec![sequence_length, hidden_dim])?;
        let hidden_states_copy = device.zeros(vec![sequence_length, hidden_dim])?;
        let hidden_states_attn_output = device.zeros(vec![sequence_length, hidden_dim])?;
        let intermediate_states = device.zeros(vec![sequence_length, intermediate_dim])?;
        let qkv_cache = device.zeros(vec![sequence_length, hidden_dim * 3])?;
        let probs = device.zeros(vec![sequence_length, vocab_size])?;
        Ok(Gpt2Context {
            input_ids,
            position_ids,
            hidden_states,
            hidden_states_copy,
            hidden_states_attn_output,
            intermediate_states,
            past_key_values,
            qkv_cache,
            probs,
        })
    }

    /// TODO
    pub fn run(&self, input_ids: Vec<usize>) -> Result<T, SmeltError> {
        let mut context = self.new_context(input_ids, self.num_heads)?;
        self.forward(&mut context)?;
        Ok(context.probs)
    }
}
