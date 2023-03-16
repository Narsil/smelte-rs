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
    pub fn forward(&self, ids: &[usize], out: &mut T) -> Result<(), SmeltError> {
        T::select(ids, &self.weight, out)
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
        let mut out = Tensor::zeros(vec![2, 2]);
        embedding.forward(&[0, 1], &mut out).unwrap();
    }

    #[test]
    fn test_embedding_errors() {
        let weights = Tensor::zeros(vec![3, 2]);
        let embedding = Embedding::new(weights);
        let mut out = Tensor::zeros(vec![2, 2]);
        assert!(embedding.forward(&[3], &mut out).is_err());
    }
}
