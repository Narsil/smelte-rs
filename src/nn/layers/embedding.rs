use crate::traits::{Tensor, TensorOps};
use crate::SmeltError;

/// TODO
#[derive(Clone)]
pub struct Embedding<T: Tensor> {
    weight: T,
}

impl<T: Tensor + TensorOps<T>> Embedding<T> {
    /// TODO
    pub fn new(weight: T) -> Self {
        Self { weight }
    }

    /// TODO
    pub fn forward(&self, ids: &[u32]) -> Result<T, SmeltError> {
        T::select(ids, &self.weight)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::f32::Tensor;

    #[test]
    fn test_embedding() {
        let weights = Tensor::zeros(vec![3, 2]);
        let embedding = Embedding::new(weights);
        let _out = embedding.forward(&[0, 1]).unwrap();
    }

    #[test]
    fn test_embedding_errors() {
        let weights = Tensor::zeros(vec![3, 2]);
        let embedding = Embedding::new(weights);
        assert!(embedding.forward(&[3]).is_err());
    }
}
