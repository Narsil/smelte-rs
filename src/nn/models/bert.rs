use crate::cpu::f32::{matmul, matmul_t, softmax, Tensor as F32Tensor};
use crate::nn::layers::{Embedding, LayerNorm, Linear};
use crate::traits::{Tensor, TensorOps};
use crate::SmeltError;

fn split_heads<'a>(q: &'a F32Tensor<'a>, num_heads: usize) -> F32Tensor<'a> {
    let sequence_length = q.shape()[0];
    let hidden_dim = q.shape()[1];
    assert_eq!(hidden_dim % num_heads, 0);
    let head_dim = hidden_dim / num_heads;
    let mut query_data = vec![0.0; num_heads * sequence_length * head_dim];
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
    F32Tensor::new(query_data, vec![num_heads, sequence_length, head_dim]).unwrap()
}

fn attention<'a>(
    query: &F32Tensor<'a>,
    key: &F32Tensor<'a>,
    value: &F32Tensor<'a>,
    num_heads: usize,
) -> F32Tensor<'a> {
    let sequence_length = query.shape()[0];
    let hidden_dim = query.shape()[1];
    assert_eq!(hidden_dim % num_heads, 0);

    let query = split_heads(query, num_heads);
    let key = split_heads(key, num_heads);
    let value = split_heads(value, num_heads);

    let mut qk = matmul_t(&query, &key).unwrap();
    let head_dim = hidden_dim / num_heads;
    let scale = (head_dim as f32).sqrt();
    qk.data_mut().iter_mut().for_each(|v| *v /= scale);

    softmax(&mut qk).unwrap();
    let out = matmul(&qk, &value).unwrap();

    let mut new_out = vec![0.0; sequence_length * hidden_dim];
    (0..num_heads).for_each(|i| {
        (0..sequence_length).for_each(|j| {
            (0..head_dim).for_each(|k| {
                let in_index = i * sequence_length * head_dim + j * head_dim + k;
                let out_index = j * hidden_dim + i * head_dim + k;
                new_out[out_index] = out.data()[in_index];
            });
        });
    });
    F32Tensor::new(new_out, vec![sequence_length, hidden_dim]).unwrap()
}

/// TODO
pub trait TensorAttention<T> {
    /// TODO
    fn attention(q: &T, k: &T, v: &T, num_heads: usize) -> Result<T, SmeltError>;
}

impl<'a> TensorAttention<F32Tensor<'a>> for F32Tensor<'a> {
    fn attention(
        q: &F32Tensor<'a>,
        k: &F32Tensor<'a>,
        v: &F32Tensor<'a>,
        num_heads: usize,
    ) -> Result<F32Tensor<'a>, SmeltError> {
        Ok(attention(q, k, v, num_heads))
    }
}

/// TODO
pub trait BertOps<T>: TensorOps<T> + TensorAttention<T> {}

impl<'a> BertOps<F32Tensor<'a>> for F32Tensor<'a> {}

/// TODO
#[derive(Clone)]
pub struct BertAttention<T: Tensor> {
    query: Linear<T>,
    key: Linear<T>,
    value: Linear<T>,
    output: Linear<T>,
    output_ln: LayerNorm<T>,
    num_heads: usize,
}

impl<T: Tensor + BertOps<T>> BertAttention<T> {
    /// TODO
    pub fn new(
        query: Linear<T>,
        key: Linear<T>,
        value: Linear<T>,
        output: Linear<T>,
        output_ln: LayerNorm<T>,
        num_heads: usize,
    ) -> Self {
        Self {
            query,
            key,
            value,
            output,
            output_ln,
            num_heads,
        }
    }

    /// TODO
    pub fn forward(&self, hidden_states: &T) -> Result<T, SmeltError> {
        let q = self.query.forward(hidden_states)?;
        let k = self.key.forward(hidden_states)?;
        let v = self.value.forward(hidden_states)?;

        let attended = T::attention(&q, &k, &v, self.num_heads)?;
        let mut tensor = self.output.forward(&attended)?;
        T::add(hidden_states, &mut tensor)?;
        self.output_ln.forward(&mut tensor)?;
        Ok(tensor)
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
    pub fn forward(&self, tensor: &T) -> Result<T, SmeltError> {
        let input_tensor = tensor.clone();
        let mut tensor = self.intermediate.forward(tensor)?;
        T::gelu(&mut tensor)?;
        let mut tensor = self.output.forward(&tensor)?;
        T::add(&input_tensor, &mut tensor)?;
        self.output_ln.forward(&mut tensor)?;
        Ok(tensor)
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
    pub fn forward(&self, tensor: &T) -> Result<T, SmeltError> {
        let tensor = self.attention.forward(tensor)?;
        self.mlp.forward(&tensor)
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
    pub fn forward(&self, tensor: &T) -> Result<T, SmeltError> {
        let mut tensor = self.layers[0].forward(tensor)?;
        for layer in &self.layers[1..] {
            tensor = layer.forward(&tensor)?;
        }
        Ok(tensor)
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
    pub fn forward(&self, input_ids: &[usize], type_ids: &[usize]) -> Result<T, SmeltError> {
        if input_ids.len() != type_ids.len() {
            return Err(SmeltError::InvalidLength {
                expected: input_ids.len(),
                got: type_ids.len(),
            });
        }

        let positions: Vec<usize> = (0..input_ids.len()).collect();

        let mut input_embeds = self.input_embeddings.forward(input_ids)?;
        let position_embeds = self.position_embeddings.forward(&positions[..])?;
        let type_embeds = self.type_embeddings.forward(type_ids)?;

        T::add(&position_embeds, &mut input_embeds)?;
        T::add(&type_embeds, &mut input_embeds)?;
        self.layer_norm.forward(&mut input_embeds)?;
        Ok(input_embeds)
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
    pub fn forward(&self, input_ids: &[usize], type_ids: &[usize]) -> Result<T, SmeltError> {
        let tensor = self.embeddings.forward(input_ids, type_ids)?;
        self.encoder.forward(&tensor)
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
    pub fn forward(&self, tensor: &T) -> Result<T, SmeltError> {
        let first = T::select(&[0], tensor)?;
        let mut tensor = self.pooler.forward(&first)?;
        T::tanh(&mut tensor)?;
        Ok(tensor)
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
    pub fn forward(&self, input_ids: &[usize], type_ids: &[usize]) -> Result<T, SmeltError> {
        let tensor = self.bert.forward(input_ids, type_ids)?;
        let tensor = self.pooler.forward(&tensor)?;
        let mut logits = self.classifier.forward(&tensor)?;
        T::softmax(&mut logits)?;
        Ok(logits)
    }
}
