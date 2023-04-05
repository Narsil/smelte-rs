use clap::Parser;
use memmap2::MmapOptions;
use safetensors::{
    tensor::{Dtype, SafeTensorError, TensorView},
    SafeTensors,
};
use serde::Deserialize;

#[cfg(feature = "cpu")]
use smelte_rs::cpu::f32::{Device, Tensor};
#[cfg(feature = "cuda")]
use smelte_rs::gpu::f32::{Device, Tensor};

use smelte_rs::nn::layers::{Embedding, LayerNorm, LinearT, UnbiasedLinear};
use smelte_rs::nn::models::gpt2::{Gpt2, Gpt2Attention, Gpt2Layer, Gpt2Model, Mlp};
use smelte_rs::SmeltError;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use thiserror::Error;
use tokenizers::Tokenizer;

#[derive(Debug, Error)]
pub enum Gpt2Error {
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
    n_head: usize,
    id2label: Option<HashMap<String, String>>,
}

impl Config {
    pub fn id2label(&self) -> Option<&HashMap<String, String>> {
        self.id2label.as_ref()
    }
}

pub fn get_label(id2label: Option<&HashMap<String, String>>, i: usize) -> Option<String> {
    let id2label: &HashMap<String, String> = id2label?;
    let label: String = id2label.get(&format!("{}", i))?.to_string();
    Some(label)
}

pub trait FromSafetensors<'a> {
    fn from_tensors(tensors: &'a SafeTensors<'a>, device: &Device) -> Self
    where
        Self: Sized;
}

fn to_tensor<'data>(view: TensorView<'data>, device: &Device) -> Result<Tensor, SmeltError> {
    let shape = view.shape().to_vec();
    let data = to_f32(view);
    #[cfg(feature = "cuda")]
    {
        Tensor::from_cpu(&data, shape, device)
    }

    #[cfg(feature = "cpu")]
    {
        Tensor::from_cpu(data, shape, device)
    }
}

pub fn to_f32(view: TensorView) -> Cow<'static, [f32]> {
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

fn linear_from<'a>(
    weights: TensorView<'a>,
    bias: TensorView<'a>,
    device: &Device,
) -> LinearT<Tensor> {
    LinearT::new(
        to_tensor(weights, device).unwrap(),
        to_tensor(bias, device).unwrap(),
    )
}

fn unbiased_linear_from<'a>(weights: TensorView<'a>, device: &Device) -> UnbiasedLinear<Tensor> {
    UnbiasedLinear::new(to_tensor(weights, device).unwrap())
}

fn linear_from_prefix<'a>(
    prefix: &str,
    tensors: &'a SafeTensors<'a>,
    device: &Device,
) -> LinearT<Tensor> {
    linear_from(
        tensors.tensor(&format!("{}.weight", prefix)).unwrap(),
        tensors.tensor(&format!("{}.bias", prefix)).unwrap(),
        device,
    )
}

fn embedding_from<'a>(weights: TensorView<'a>, device: &Device) -> Embedding<Tensor> {
    Embedding::new(to_tensor(weights, device).unwrap())
}

impl<'a> FromSafetensors<'a> for Gpt2<Tensor> {
    fn from_tensors(tensors: &'a SafeTensors<'a>, device: &Device) -> Self
    where
        Self: Sized,
    {
        let wte = embedding_from(tensors.tensor("wte.weight").unwrap(), device);
        let wpe = embedding_from(tensors.tensor("wpe.weight").unwrap(), device);
        let h = Gpt2Model::from_tensors(tensors, device);
        let ln_f = layer_norm_from_prefix("ln_f", &tensors, device);
        let lm_head = unbiased_linear_from(tensors.tensor("wte.weight").unwrap(), device);
        // TODO number of heads
        Gpt2::new(wte, wpe, h, ln_f, lm_head, 12)
    }
}

fn gpt2_layer_from_tensors<'a>(
    index: usize,
    tensors: &'a SafeTensors<'a>,
    device: &Device,
) -> Gpt2Layer<Tensor> {
    let ln_1 = layer_norm_from_prefix(&format!("h.{index}.ln_1"), tensors, device);
    let ln_2 = layer_norm_from_prefix(&format!("h.{index}.ln_2"), tensors, device);
    let attention = gpt2_attention_from_tensors(index, tensors, device);
    let mlp = gpt2_mlp_from_tensors(index, tensors, device);
    Gpt2Layer::new(attention, mlp, ln_1, ln_2)
}
fn gpt2_attention_from_tensors<'a>(
    index: usize,
    tensors: &'a SafeTensors<'a>,
    device: &Device,
) -> Gpt2Attention<Tensor> {
    let c_attn = linear_from_prefix(&format!("h.{index}.attn.c_attn"), tensors, device);
    let c_proj = linear_from_prefix(&format!("h.{index}.attn.c_proj"), tensors, device);
    Gpt2Attention::new(c_attn, c_proj)
}

fn gpt2_mlp_from_tensors<'a>(
    index: usize,
    tensors: &'a SafeTensors<'a>,
    device: &Device,
) -> Mlp<Tensor> {
    let c_fc = linear_from_prefix(&format!("h.{index}.mlp.c_fc"), tensors, device);
    let c_proj = linear_from_prefix(&format!("h.{index}.mlp.c_proj"), tensors, device);
    Mlp::new(c_fc, c_proj)
}

fn layer_norm_from_prefix<'a>(
    prefix: &str,
    tensors: &'a SafeTensors<'a>,
    device: &Device,
) -> LayerNorm<Tensor> {
    let epsilon = 1e-5;
    if let (Ok(weight), Ok(bias)) = (
        tensors.tensor(&format!("{}.weight", prefix)),
        tensors.tensor(&format!("{}.bias", prefix)),
    ) {
        LayerNorm::new(
            to_tensor(weight, device).unwrap(),
            to_tensor(bias, device).unwrap(),
            epsilon,
        )
    } else {
        LayerNorm::new(
            to_tensor(
                tensors.tensor(&format!("{}.gamma", prefix)).unwrap(),
                device,
            )
            .unwrap(),
            to_tensor(tensors.tensor(&format!("{}.beta", prefix)).unwrap(), device).unwrap(),
            epsilon,
        )
    }
}

impl<'a> FromSafetensors<'a> for Gpt2Model<Tensor> {
    fn from_tensors(tensors: &'a SafeTensors<'a>, device: &Device) -> Self
    where
        Self: Sized,
    {
        // TODO ! Count heads from tensors present
        let layers: Vec<_> = (0..12)
            .map(|i| gpt2_layer_from_tensors(i, tensors, device))
            .collect();
        Self::new(layers)
    }
}

#[derive(Parser)]
struct Args {
    /// Prompt to run
    #[arg(short, long, default_value_t = String::from("Stocks rallied and the British pound gained"))]
    prompt: String,
    /// Number of times to run the prompt
    #[arg(short, long, default_value_t = 1)]
    number: u8,
}

pub fn run() -> Result<(), Gpt2Error> {
    let start = std::time::Instant::now();
    let args = Args::parse();
    let string = args.prompt;
    let n = args.number;

    let model_id = "Narsil/fast_gpt2";

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

    #[cfg(feature = "cuda")]
    let device = Device::new(0).unwrap();

    #[cfg(feature = "cpu")]
    let device = Device {};

    let mut gpt2 = Gpt2::from_tensors(&tensors, &device);
    gpt2.set_num_heads(config.n_head);

    println!("Loaded {:?}", start.elapsed());

    let encoded = tokenizer.encode(string.clone(), false).unwrap();
    let encoded = tokenizer.post_process(encoded, None, true).unwrap();

    println!("Loaded & encoded {:?}", start.elapsed());

    for _ in 0..n {
        println!("Running gpt2 inference on {string:?}");
        let inference_start = std::time::Instant::now();
        let input_ids: Vec<_> = encoded.get_ids().iter().map(|i| *i as usize).collect();
        let probs = gpt2.run(input_ids).unwrap();

        let id2label = config.id2label();
        let mut outputs: Vec<_> = probs
            .cpu_data()
            .unwrap()
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

fn main() {
    #[cfg(not(any(feature = "cuda", feature = "cpu")))]
    unreachable!("Requires cuda/cpu feature");

    #[cfg(any(feature = "cuda", feature = "cpu"))]
    run().unwrap()
}
