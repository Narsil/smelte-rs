use memmap2::MmapOptions;
use safetensors::{
    tensor::{Dtype, SafeTensorError, TensorView},
    SafeTensors,
};
use serde::Deserialize;
use smelte-rs::cpu::f32::Tensor;
use smelte-rs::nn::layers::{Embedding, LayerNorm, Linear};
use smelte-rs::nn::models::bert::{
    Bert, BertAttention, BertClassifier, BertEmbeddings, BertEncoder, BertLayer, BertPooler, Mlp,
};
use smelte-rs::TensorError;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use thiserror::Error;
use tokenizers::Tokenizer;

#[derive(Debug, Error)]
pub enum BertError {
    #[error("i/o error")]
    IOError(#[from] std::io::Error),
    #[error("safetensor error")]
    SafeTensorError(#[from] SafeTensorError),
    #[error("slice error")]
    Slice(#[from] std::array::TryFromSliceError),
    #[error("parsing int error")]
    ParseIntError(#[from] core::num::ParseIntError),
    #[error("JSON parsing error")]
    JSONError(#[from] serde_json::Error),
}

#[derive(Clone, Deserialize)]
pub struct Config {
    num_attention_heads: usize,
    id2label: Option<HashMap<String, String>>,
}

impl Config {
    pub fn id2label(&self) -> Option<&HashMap<String, String>> {
        self.id2label.as_ref()
    }
}

pub fn main() -> Result<(), BertError> {
    let start = std::time::Instant::now();
    let args: Vec<String> = std::env::args().collect();
    let (n, string) = if args.len() > 1 {
        let mut string = "".to_string();
        let mut n = 1;
        let mut i = 1;
        while i < args.len() {
            if args[i] == "-n" {
                i += 1;
                n = args[i].parse().unwrap();
            } else if args[i] == "-h" {
                println!(
                    "Use `-n 3` to run the prompt n times, the rest is interpreted as the prompt."
                );
                return Ok(());
            } else {
                string.push_str(&args[i]);
                i += 1;
            }
        }
        (n, string)
    } else {
        (
            1,
            "Stocks rallied and the British pound gained.".to_string(),
        )
    };

    let model_id = "Narsil/finbert";

    let model_id_slug = model_id.replace('/', "-");

    let filename = format!("model-{model_id_slug}.safetensors");
    if !std::path::Path::new(&filename).exists() {
        println!(
            r#"Model not found, try downloading it with \n
        `curl https://huggingface.co/{model_id}/resolve/main/model.safetensors -o model-{model_id_slug}.safetensors -L`
        `curl https://huggingface.co/{model_id}/resolve/main/tokenizer.json -o tokenizer-{model_id_slug}.json -L`
        `curl https://huggingface.co/{model_id}/resolve/main/config.json -o config-{model_id_slug}.json -L`
        "#
        );
    }

    let file = File::open(filename)?;
    let buffer = unsafe { MmapOptions::new().map(&file)? };
    let tensors = SafeTensors::deserialize(&buffer)?;
    println!("Safetensors {:?}", start.elapsed());

    let filename = format!("tokenizer-{model_id_slug}.json");
    if !std::path::Path::new(&filename).exists() {
        println!(
            r#"Tokenizer not found, try downloading it with \n
        `curl https://huggingface.co/{model_id}/resolve/main/tokenizer.json -o tokenizer-{model_id_slug}.json -L`
        "#
        );
    }
    let tokenizer = Tokenizer::from_file(filename).unwrap();
    println!("Tokenizer {:?}", start.elapsed());

    let filename = format!("config-{model_id_slug}.json");
    if !std::path::Path::new(&filename).exists() {
        println!(
            r#"Config not found, try downloading it with \n
        `curl https://huggingface.co//resolve/main/config.json -o config-{model_id_slug}.json -L`
        "#
        );
    }
    let config_str: String = std::fs::read_to_string(filename).expect("Could not read config");
    let config: Config = serde_json::from_str(&config_str).expect("Could not parse Config");

    let bert = BertClassifier::from_tensors(&tensors);
    println!("Loaded {:?}", start.elapsed());

    let encoded = tokenizer.encode(string.clone(), false).unwrap();
    let encoded = tokenizer.post_process(encoded, None, true).unwrap();

    println!("Loaded & encoded {:?}", start.elapsed());

    let input_ids: Vec<_> = encoded.get_ids().iter().map(|i| *i as usize).collect();
    let position_ids: Vec<_> = (0..input_ids.len()).collect();
    let type_ids: Vec<_> = encoded.get_type_ids().iter().map(|i| *i as usize).collect();
    let mut context = bert.new_context(
        input_ids,
        position_ids,
        type_ids,
        config.num_attention_heads,
    );
    for _ in 0..n {
        println!("Running bert inference on `{string:?}`");
        let inference_start = std::time::Instant::now();
        bert.forward(&mut context).unwrap();

        let probs = context.probs();

        let id2label = config.id2label();
        let mut outputs: Vec<_> = probs
            .data()
            .iter()
            .enumerate()
            .map(|(i, &p)| (get_label(id2label, i).unwrap_or(format!("LABEL_{}", i)), p))
            .collect();
        outputs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        println!("Probs {:?}", outputs);
        println!("Inference in {:?}", inference_start.elapsed());
    }
    println!("Total Inference {:?}", start.elapsed());
    Ok(())
}

pub fn get_label(id2label: Option<&HashMap<String, String>>, i: usize) -> Option<String> {
    let id2label: &HashMap<String, String> = id2label?;
    let label: String = id2label.get(&format!("{}", i))?.to_string();
    Some(label)
}

pub trait FromSafetensors<'a> {
    fn from_tensors(tensors: &'a SafeTensors<'a>) -> Self
    where
        Self: Sized;
}

fn to_tensor<'data>(view: TensorView<'data>) -> Result<Tensor<'data>, TensorError> {
    let shape = view.shape().to_vec();
    let data = to_f32(view);
    Tensor::new(data, shape)
}

pub fn to_f32<'data>(view: TensorView<'data>) -> Cow<'data, [f32]> {
    assert_eq!(view.dtype(), Dtype::F32);
    let v = view.data();
    if (v.as_ptr() as usize) % 4 == 0 {
        // SAFETY This is safe because we just checked that this
        // was correctly aligned.
        let data: &[f32] =
            unsafe { std::slice::from_raw_parts(v.as_ptr() as *const f32, v.len() / 4) };
        Cow::Borrowed(data)
    } else {
        let mut c = Vec::with_capacity(v.len() / 4);
        let mut i = 0;
        while i < v.len() {
            c.push(f32::from_le_bytes([v[i], v[i + 1], v[i + 2], v[i + 3]]));
            i += 4;
        }
        Cow::Owned(c)
    }
}

fn linear_from<'a>(weights: TensorView<'a>, bias: TensorView<'a>) -> Linear<Tensor<'a>> {
    Linear::new(to_tensor(weights).unwrap(), to_tensor(bias).unwrap())
}

fn linear_from_prefix<'a>(prefix: &str, tensors: &'a SafeTensors<'a>) -> Linear<Tensor<'a>> {
    linear_from(
        tensors.tensor(&format!("{}.weight", prefix)).unwrap(),
        tensors.tensor(&format!("{}.bias", prefix)).unwrap(),
    )
}

fn embedding_from<'a>(weights: TensorView<'a>) -> Embedding<Tensor<'a>> {
    Embedding::new(to_tensor(weights).unwrap())
}

impl<'a> FromSafetensors<'a> for BertClassifier<Tensor<'a>> {
    fn from_tensors(tensors: &'a SafeTensors<'a>) -> Self
    where
        Self: Sized,
    {
        let pooler = BertPooler::from_tensors(tensors);
        let bert = Bert::from_tensors(tensors);
        let (weight, bias) = if let (Ok(weight), Ok(bias)) = (
            tensors.tensor("classifier.weight"),
            tensors.tensor("classifier.bias"),
        ) {
            (weight, bias)
        } else {
            (
                tensors.tensor("cls.seq_relationship.weight").unwrap(),
                tensors.tensor("cls.seq_relationship.bias").unwrap(),
            )
        };
        let classifier = linear_from(weight, bias);
        Self::new(bert, pooler, classifier)
    }
}

impl<'a> FromSafetensors<'a> for BertPooler<Tensor<'a>> {
    fn from_tensors(tensors: &'a SafeTensors<'a>) -> Self
    where
        Self: Sized,
    {
        let pooler = linear_from(
            tensors.tensor("bert.pooler.dense.weight").unwrap(),
            tensors.tensor("bert.pooler.dense.bias").unwrap(),
        );
        Self::new(pooler)
    }
}

impl<'a> FromSafetensors<'a> for Bert<Tensor<'a>> {
    fn from_tensors(tensors: &'a SafeTensors<'a>) -> Self
    where
        Self: Sized,
    {
        let embeddings = BertEmbeddings::from_tensors(tensors);
        let encoder = BertEncoder::from_tensors(tensors);
        Bert::new(embeddings, encoder)
    }
}

impl<'a> FromSafetensors<'a> for BertEmbeddings<Tensor<'a>> {
    fn from_tensors(tensors: &'a SafeTensors<'a>) -> Self
    where
        Self: Sized,
    {
        let input_embeddings = embedding_from(
            tensors
                .tensor("bert.embeddings.word_embeddings.weight")
                .unwrap(),
        );
        let position_embeddings = embedding_from(
            tensors
                .tensor("bert.embeddings.position_embeddings.weight")
                .unwrap(),
        );
        let type_embeddings = embedding_from(
            tensors
                .tensor("bert.embeddings.token_type_embeddings.weight")
                .unwrap(),
        );

        let layer_norm = layer_norm_from_prefix("bert.embeddings.LayerNorm", &tensors);
        BertEmbeddings::new(
            input_embeddings,
            position_embeddings,
            type_embeddings,
            layer_norm,
        )
    }
}

fn bert_layer_from_tensors<'a>(
    index: usize,
    tensors: &'a SafeTensors<'a>,
) -> BertLayer<Tensor<'a>> {
    let attention = bert_attention_from_tensors(index, tensors);
    let mlp = bert_mlp_from_tensors(index, tensors);
    BertLayer::new(attention, mlp)
}
fn bert_attention_from_tensors<'a>(
    index: usize,
    tensors: &'a SafeTensors<'a>,
) -> BertAttention<Tensor<'a>> {
    let query = linear_from_prefix(
        &format!("bert.encoder.layer.{index}.attention.self.query"),
        tensors,
    );
    let key = linear_from_prefix(
        &format!("bert.encoder.layer.{index}.attention.self.key"),
        tensors,
    );
    let value = linear_from_prefix(
        &format!("bert.encoder.layer.{index}.attention.self.value"),
        tensors,
    );
    let output = linear_from_prefix(
        &format!("bert.encoder.layer.{index}.attention.output.dense"),
        tensors,
    );
    let output_ln = layer_norm_from_prefix(
        &format!("bert.encoder.layer.{index}.attention.output.LayerNorm"),
        &tensors,
    );
    BertAttention::new(query, key, value, output, output_ln)
}

fn bert_mlp_from_tensors<'a>(index: usize, tensors: &'a SafeTensors<'a>) -> Mlp<Tensor<'a>> {
    let intermediate = linear_from_prefix(
        &format!("bert.encoder.layer.{index}.intermediate.dense"),
        tensors,
    );
    let output = linear_from_prefix(&format!("bert.encoder.layer.{index}.output.dense"), tensors);
    let output_ln = layer_norm_from_prefix(
        &format!("bert.encoder.layer.{index}.output.LayerNorm"),
        &tensors,
    );
    Mlp::new(intermediate, output, output_ln)
}

fn layer_norm_from_prefix<'a>(prefix: &str, tensors: &'a SafeTensors<'a>) -> LayerNorm<Tensor<'a>> {
    let epsilon = 1e-5;
    if let (Ok(weight), Ok(bias)) = (
        tensors.tensor(&format!("{}.weight", prefix)),
        tensors.tensor(&format!("{}.bias", prefix)),
    ) {
        LayerNorm::new(
            to_tensor(weight).unwrap(),
            to_tensor(bias).unwrap(),
            epsilon,
        )
    } else {
        LayerNorm::new(
            to_tensor(tensors.tensor(&format!("{}.gamma", prefix)).unwrap()).unwrap(),
            to_tensor(tensors.tensor(&format!("{}.beta", prefix)).unwrap()).unwrap(),
            epsilon,
        )
    }
}
impl<'a> FromSafetensors<'a> for BertEncoder<Tensor<'a>> {
    fn from_tensors(tensors: &'a SafeTensors<'a>) -> Self
    where
        Self: Sized,
    {
        // TODO ! Count heads from tensors present
        let layers: Vec<_> = (0..12)
            .map(|i| bert_layer_from_tensors(i, tensors))
            .collect();
        Self::new(layers)
    }
}
