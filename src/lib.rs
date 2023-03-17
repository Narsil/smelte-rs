#![deny(missing_docs)]
//! # What is smelte-rs ?
//!
//! Smelt is a ML library focusing on inference, small depedencies with as many optimizations
//! as possible, and still be readable and easy to use.
//!
//! Keep unsafe usage limited and only for performance.
//!
//! # Running models
//!
//! Try running Bert on text classification example.
//!
//! ```bash
//! # Download the model + tokenizer + config
//! # This is a clone of https://huggingface.co/ProsusAI/finbert with safetensors support.
//! curl https://huggingface.co/Narsil/finbert/resolve/main/model.safetensors -o model-Narsil-finbert.safetensors -L
//! curl https://huggingface.co/Narsil/finbert/resolve/main/tokenizer.json -o tokenizer-Narsil-finbert.json -L
//! curl https://huggingface.co/Narsil/finbert/resolve/main/config.json -o config-Narsil-finbert.json -L
//!
//! # Linux
//! cargo run --example bert --release --features intel-mkl -- "This is a test" -n 3
//!
//! # M1
//! cargo run --example bert --release -- "This is a test" -n 3
//! ```
//!
//! # Why not use library X ?
//!
//! Many other libraries for ML out there, torch and tensorflow are great but
//! are now extremely heavy with no option to statically link against.
//! Libraries like ONNX are great too, but when an operator is missing out, it's
//! really hard to work against.
//!
//! For low level libraries. [ggml](https://github.com/ggerganov/ggml) is a great
//! library, no dependencies, extremely small binary size. It's actually an
//! inspiration for this project ! But I'm not good enough a C++ programmer to hack it
//! efficiently enough. Also it's hard to use outside of the intended scope, for
//! instance when writing a webserver/API, or if we wanted to use CUDA as a backend.
//!
//! [dfdx](https://github.com/coreylowman/dfdx) is another super nice project.
//! I drew inspiration from it too. The problem with dfdx was the typing system
//! which while extremely powerful (compile time size checking) it was getting
//! in the way of getting things done, and optimizing for it is not as trivial as
//! it's harder to know what's going on.
//!
//! # Important missing features
//!
//! - [ ] GPU support
//! - [ ] f16 support
//! - [ ] q4_0 support (akin to ggml)
//! - [ ] remove blas/intel-mkl dependency (write the intrinsics ourselves)
//! - [ ] Add more models
//!   - [ ] Roberta for <https://huggingface.co/roberta-base-openai-detector>
//!   - [ ] Distilbert
//!   - [ ] Gpt2 (Almost done)
//!   - [ ] Llama
//!
//! # The architecture of this library:
//!
//! - [cpu] is containing all the various precisions backend operations, tensor structs.
//!   This is your go-to if you want to code everything from scratch.
//! - [nn] contains all the basic layers, and actual model implementations. Code should
//! look closely like torch implementations.
//! - [traits] Contains the glue that allows [nn] to be written independantly of [cpu]
//!   which should hopefully making using different precisions (or backends) quite easy.
//!
//!
//! # How does the model look like:
//!
//! ```ignore
//! pub struct BertClassifier<T: Tensor + TensorOps<T>> {
//!     bert: Bert<T>,
//!     pooler: BertPooler<T>,
//!     classifier: Linear<T>,
//! }
//!
//! impl<T: Tensor + TensorOps<T>> BertClassifier<T> {
//!     pub fn new(bert: Bert<T>, pooler: BertPooler<T>, classifier: Linear<T>) -> Self {
//!         Self {
//!             bert,
//!             pooler,
//!             classifier,
//!         }
//!     }
//!     pub fn forward(&self, input_ids: &[usize], type_ids: &[usize]) -> Result<T, SmeltError> {
//!         let tensor = self.bert.forward(input_ids, type_ids)?;
//!         let tensor = self.pooler.forward(&tensor)?;
//!         let mut logits = self.classifier.forward(&tensor)?;
//!         T::softmax(&mut logits)?;
//!         Ok(logits)
//!     }
//! }
//! ```
//!
//! # What's the performance like ?
//!
//! On a relatively old computer (i7-4790 CPU) This gives ~40ms/token for GPT-2
//! in full f32 precision.
//! For comparison, on the same hardware `torch` gives ~47ms/token and ggml ~37ms.
//!
//! Current implementations does *not* use threading, nor precomputed gelu/exp
//! nor f16 shortcuts that ggml can use (like for the softmax).
//!
//! So there is still lots of room for improvement, and most of the current performance
//! comes from using `intel-mkl` library, which can be dropped once this implements
//! the various ops from ggml (hopefully to get the full performance).

/// The various CPU implementations
pub mod cpu;

/// The neural networks
pub mod nn;

/// The traits for generic implementations
pub mod traits;

/// Error linked to the tensor creation
#[derive(Debug)]
pub enum TensorError {
    /// The arguments to the tensor creation are invalid, the shape doesn't match
    /// the size of the buffer.
    InvalidBuffer {
        /// The size of the buffer sent
        buffer_size: usize,
        /// The shape of the tensor to create
        shape: Vec<usize>,
    },
}

/// Potential errors when using the library
#[derive(Debug)]
pub enum SmeltError {
    /// The operation could not succeed because the shapes are not valid.
    DimensionMismatch {
        /// The shape that we should have seen
        expected: Vec<usize>,
        /// The shape that we received
        got: Vec<usize>,
    },
    /// The tensor given has insufficient rank (rank 2 means a tensor that has a shape of length 2)
    InsufficientRank {
        /// The minimum rank that we expect
        minimum_rank: usize,
    },
    /// The tensor given has not the expected rank (rank 2 means a tensor that has a shape of length 2)
    InvalidRank {
        /// The rank that we expect
        expected_rank: usize,
    },
    /// The tensor given has not enough room for the operations
    VectorTooSmall {
        /// The minimum size that we expect
        minimum: usize,
    },

    /// The select operation attempted to select out of the tensor
    OutOfVocabulary {
        /// The vocabulary size
        vocab_size: usize,
        /// culprit id
        id: usize,
    },

    /// Some slices do not have the expected lengths
    InvalidLength {
        /// The size we expected
        expected: usize,
        /// The size we got
        got: usize,
    },
}

#[cfg(test)]
mod tests {
    pub(crate) fn simplify(data: &[f32]) -> Vec<f32> {
        let precision = 3;
        let m = 10.0 * 10.0f32.powf(precision as f32);
        data.iter().map(|x| (x * m).round() / m).collect()
    }

    // fn assert_float_eq(left: &[f32], right: &[f32]) {
    //     assert_eq!(left.len(), right.len());

    //     left.iter().zip(right.iter()).for_each(|(l, r)| {
    //         assert!(
    //             (l - r).abs() / l.abs() < 1e-4,
    //             "{l} != {r}\n{left:?}\n{right:?}"
    //         );
    //     });
    // }
}
