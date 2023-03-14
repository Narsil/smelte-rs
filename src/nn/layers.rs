use crate::traits::{Tensor, TensorAdd, TensorMatmul};

/// Linear layer, applies matmul(x, W) + b
#[derive(Clone)]
pub struct Linear<T: Tensor> {
    weight: T,
    bias: T,
}

impl<T: Tensor + TensorMatmul<T> + TensorAdd<T>> Linear<T> {
    /// Linear layer creation
    pub fn new(weight: T, bias: T) -> Self {
        Self { weight, bias }
    }

    /// Forward pass
    pub fn forward(&self, tensor: &mut T) {
        let m = tensor.shape()[0];
        let n = self.weight.shape()[0];
        let mut c = T::zeros(vec![m, n]);

        T::matmul_t(tensor, &self.weight, &mut c).unwrap();
        T::add(&self.bias, &mut c).unwrap();
        *tensor = c;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::f32::Tensor;

    #[test]
    fn test_linear() {
        let mut zeros = Tensor::zeros(vec![2, 2]);
        let weights = Tensor::zeros(vec![3, 2]);
        let bias = Tensor::zeros(vec![3]);

        let linear = Linear::new(weights, bias);

        linear.forward(&mut zeros)
    }
}
