use crate::cpu::f32::{matmul, matmul_t, softmax, Tensor as F32Tensor};
use crate::nn::layers::{Embedding, LayerNorm, Linear};
use crate::traits::{Tensor, TensorOps};
use crate::SmeltError;

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
    // Store the q splitted_heads
    q_cache: T,
    // Store the k splitted_heads
    k_cache: T,
    // Store the k splitted_heads
    v_cache: T,
    // Store the qk result
    qk: T,
    // Intermediate states (H, 4H)
    intermediate_states: T,
    pool: T,
    pool_output: T,
    logits: T,
}

fn split_heads(q: &F32Tensor, out_q: &mut F32Tensor) -> Result<(), SmeltError> {
    let num_heads = out_q.shape()[0];
    let sequence_length = out_q.shape()[1];
    let head_dim = out_q.shape()[1];
    let hidden_dim = head_dim * num_heads;

    let query_data = out_q.data_mut();
    (0..num_heads).for_each(|i| {
        (0..sequence_length).for_each(|j| {
            (0..head_dim).for_each(|k| {
                let index = j * hidden_dim + i * head_dim + k;
                let out_index = i * sequence_length * head_dim + j * head_dim + k;
                let value = q.data()[index];
                query_data[out_index] = value;
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
    k_weights.forward(&ctx.hidden_states, &mut ctx.hidden_states_copy)?;
    split_heads(&ctx.hidden_states_copy, &mut ctx.k_cache)?;
    v_weights.forward(&ctx.hidden_states, &mut ctx.hidden_states_copy)?;
    split_heads(&ctx.hidden_states_copy, &mut ctx.v_cache)?;

    matmul_t(&ctx.q_cache, &ctx.k_cache, &mut ctx.qk).unwrap();

    let num_heads = ctx.qk.shape()[0];
    let sequence_length = ctx.qk.shape()[1];
    let head_dim = ctx.qk.shape()[2];
    let hidden_dim = head_dim * num_heads;
    let scale = (head_dim as f32).sqrt();
    ctx.qk.data_mut().iter_mut().for_each(|v| *v /= scale);

    softmax(&mut ctx.qk).unwrap();
    matmul(&ctx.qk, &ctx.q_cache, &mut ctx.hidden_states_copy).unwrap();

    let new_out = &mut ctx.hidden_states_attn_output.data_mut();
    (0..num_heads).for_each(|i| {
        (0..sequence_length).for_each(|j| {
            (0..head_dim).for_each(|k| {
                let in_index = i * sequence_length * head_dim + j * head_dim + k;
                let out_index = j * hidden_dim + i * head_dim + k;
                new_out[out_index] = (ctx.hidden_states_copy).data()[in_index];
            });
        });
    });
    Ok(())
}

/// TODO
pub trait TensorAttention<T: Tensor> {
    /// TODO
    fn attention(
        q: &Linear<T>,
        k: &Linear<T>,
        v: &Linear<T>,
        ctx: &mut BertContext<T>,
    ) -> Result<(), SmeltError>;
}

impl<'a> TensorAttention<F32Tensor<'a>> for F32Tensor<'a> {
    fn attention(
        q: &Linear<F32Tensor<'a>>,
        k: &Linear<F32Tensor<'a>>,
        v: &Linear<F32Tensor<'a>>,
        ctx: &mut BertContext<F32Tensor<'a>>,
    ) -> Result<(), SmeltError> {
        attention(q, k, v, ctx)?;
        Ok(())
    }
}

/// TODO
pub trait BertOps<T: Tensor>: TensorOps<T> + TensorAttention<T> {}

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
        self.intermediate
            .forward(&ctx.hidden_states, &mut ctx.intermediate_states)?;
        T::gelu(&mut ctx.intermediate_states)?;
        self.output
            .forward(&ctx.intermediate_states, &mut ctx.hidden_states_copy)?;
        T::add(&ctx.hidden_states_copy, &mut ctx.hidden_states)?;
        self.output_ln.forward(&mut ctx.hidden_states)?;
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
        self.attention.forward(ctx)?;
        self.mlp.forward(ctx)
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
        self.position_embeddings
            .forward(position_ids, &mut ctx.hidden_states_copy)?;
        T::add(&ctx.hidden_states_copy, &mut ctx.hidden_states)?;

        self.type_embeddings
            .forward(type_ids, &mut ctx.hidden_states_copy)?;
        T::add(&ctx.hidden_states_copy, &mut ctx.hidden_states)?;
        self.layer_norm.forward(&mut ctx.hidden_states)?;
        Ok(())
    }
}

/// TODO
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
pub struct BertClassifier<T: Tensor + BertOps<T>> {
    bert: Bert<T>,
    pooler: BertPooler<T>,
    classifier: Linear<T>,
}

impl<T: Tensor + BertOps<T> + TensorAttention<T>> BertClassifier<T> {
    /// TODO
    pub fn new(bert: Bert<T>, pooler: BertPooler<T>, classifier: Linear<T>) -> Self {
        Self {
            bert,
            pooler,
            classifier,
        }
    }

    /// TODO
    pub fn forward(&self, ctx: &mut BertContext<T>) -> Result<(), SmeltError> {
        self.bert.forward(ctx)?;
        self.pooler.forward(ctx)?;
        self.classifier.forward(&ctx.pool_output, &mut ctx.logits)?;
        T::softmax(&mut ctx.logits)?;
        Ok(())
    }
}
