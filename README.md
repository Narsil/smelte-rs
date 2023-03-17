[![Crates.io](https://img.shields.io/crates/v/smelt.svg)](https://crates.io/crates/smelt)
[![Documentation](https://docs.rs/smelt/badge.svg)](https://docs.rs/smelt/)
[![Codecov](https://codecov.io/github/Narsil/smelt/coverage.svg?branch=main)](https://codecov.io/gh/Narsil/smelt)
[![Dependency status](https://deps.rs/repo/github/Narsil/smelt/status.svg)](https://deps.rs/repo/github/Narsil/smelt)

# smelt

## What is smelt ?

Smelt is a ML library focusing on inference, small depedencies with as many optimizations
as possible, and still be readable and easy to use.

Keep unsafe usage limited and only for performance.

## Running models

Try running Bert on text classification example.

```bash
# Download the model + tokenizer + config
# This is a clone of https://huggingface.co/ProsusAI/finbert with safetensors support.
curl https://huggingface.co/Narsil/finbert/resolve/main/model.safetensors -o model-Narsil-finbert.safetensors -L
curl https://huggingface.co/Narsil/finbert/resolve/main/tokenizer.json -o tokenizer-Narsil-finbert.json -L
curl https://huggingface.co/Narsil/finbert/resolve/main/config.json -o config-Narsil-finbert.json -L

# Linux
cargo run --example bert --release --features intel-mkl -- "This is a test" -n 3

# M1
cargo run --example bert --release -- "This is a test" -n 3
```

## Why not use library X ?

Many other libraries for ML out there, torch and tensorflow are great but
are now extremely heavy with no option to statically link against.
Libraries like ONNX are great too, but when an operator is missing out, it's
really hard to work against.

For low level libraries. [ggml](https://github.com/ggerganov/ggml) is a great
library, no dependencies, extremely small binary size. It's actually an
inspiration for this project ! But I'm not good enough a C++ programmer to hack it
efficiently enough. Also it's hard to use outside of the intended scope, for
instance when writing a webserver/API, or if we wanted to use CUDA as a backend.

[dfdx](https://github.com/coreylowman/dfdx) is another super nice project.
I drew inspiration from it too. The problem with dfdx was the typing system
which while extremely powerful (compile time size checking) it was getting
in the way of getting things done, and optimizing for it is not as trivial as
it's harder to know what's going on.

## The architecture of this library:

- [cpu] is containing all the various precisions backend operations, tensor structs.
  This is your go-to if you want to code everything from scratch.
- [nn] contains all the basic layers, and actual model implementations. Code should
look closely like torch implementations.
- [traits] Contains the glue that allows [nn] to be written independantly of [cpu]
  which should hopefully making using different precisions (or backends) quite easy.


## How does the model look like:

```rust
pub struct BertClassifier<T: Tensor + TensorOps<T>> {
    bert: Bert<T>,
    pooler: BertPooler<T>,
    classifier: Linear<T>,
}

impl<T: Tensor + TensorOps<T>> BertClassifier<T> {
    pub fn new(bert: Bert<T>, pooler: BertPooler<T>, classifier: Linear<T>) -> Self {
        Self {
            bert,
            pooler,
            classifier,
        }
    }
    pub fn forward(&self, input_ids: &[usize], type_ids: &[usize]) -> Result<T, SmeltError> {
        let tensor = self.bert.forward(input_ids, type_ids)?;
        let tensor = self.pooler.forward(&tensor)?;
        let mut logits = self.classifier.forward(&tensor)?;
        T::softmax(&mut logits)?;
        Ok(logits)
    }
}
```

## What's the performance like ?

On a relatively old computer (i7-4790 CPU) This gives ~40ms/token for GPT-2
in full f32 precision.
For comparison, on the same hardware `torch` gives ~47ms/token and ggml ~37ms.

Current implementations does *not* use threading, nor precomputed gelu/exp
nor f16 shortcuts that ggml can use (like for the softmax).

So there is still lots of room for improvement, and most of the current performance
comes from using `intel-mkl` library, which can be dropped once this implements
the various ops from ggml (hopefully to get the full performance).
