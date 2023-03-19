use crate::traits::{Tensor, TensorOps};
use crate::SmeltError;

/// TODO
#[derive(Clone)]
pub struct LayerNorm<T: Tensor> {
    weight: T,
    bias: T,
    epsilon: f32,
}

impl<T: Tensor + TensorOps<T>> LayerNorm<T> {
    /// TODO
    pub fn new(weight: T, bias: T, epsilon: f32) -> Self {
        Self {
            weight,
            bias,
            epsilon,
        }
    }

    /// TODO
    pub fn forward(&self, tensor: &mut T) -> Result<(), SmeltError> {
        T::normalize(tensor, self.epsilon)?;
        T::broadcast_mul(&self.weight, tensor)?;
        T::broadcast_add(&self.bias, tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::f32::Tensor;

    #[test]
    fn test_layer_norm() {
        let mut zeros = Tensor::zeros(vec![3, 2]);
        let weights = Tensor::zeros(vec![3, 2]);
        let bias = Tensor::zeros(vec![2]);

        let linear = LayerNorm::new(weights, bias, 1e-5);

        linear.forward(&mut zeros).unwrap();
    }
}
