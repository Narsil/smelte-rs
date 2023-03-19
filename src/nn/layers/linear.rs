use crate::traits::{Tensor, TensorOps};
use crate::SmeltError;

/// Linear layer, applies matmul(x, W.T) + b
#[derive(Clone)]
pub struct Linear<T: Tensor> {
    weight: T,
    bias: T,
}

impl<T: Tensor + TensorOps<T>> Linear<T> {
    /// Linear layer creation
    pub fn new(weight: T, bias: T) -> Self {
        Self { weight, bias }
    }

    /// Forward pass
    pub fn forward(&self, tensor: &T, out: &mut T) -> Result<(), SmeltError> {
        T::matmul_t(tensor, &self.weight, out)?;
        T::broadcast_add(&self.bias, out)?;
        Ok(())
    }

    /// TODO
    pub fn weight(&self) -> &T {
        &self.weight
    }

    /// TODO
    pub fn bias(&self) -> &T {
        &self.bias
    }
}

/// Linear layer, applies matmul(x, W) + b (also named conv1d sometimes)
#[derive(Clone)]
pub struct LinearT<T: Tensor> {
    weight: T,
    bias: T,
}

impl<T: Tensor + TensorOps<T>> LinearT<T> {
    /// LinearT layer creation
    pub fn new(weight: T, bias: T) -> Self {
        Self { weight, bias }
    }

    /// Forward pass
    pub fn forward(&self, tensor: &T, out: &mut T) -> Result<(), SmeltError> {
        T::matmul_t(tensor, &self.weight, out)?;
        T::broadcast_add(&self.bias, out)?;
        Ok(())
    }
}

/// UnbiasedLinear layer, applies matmul(x, W.T)
#[derive(Clone)]
pub struct UnbiasedLinear<T: Tensor> {
    weight: T,
}

impl<T: Tensor + TensorOps<T>> UnbiasedLinear<T> {
    /// UnbiasedLinear layer creation
    pub fn new(weight: T) -> Self {
        Self { weight }
    }

    /// Forward pass
    pub fn forward(&self, tensor: &T, out: &mut T) -> Result<(), SmeltError> {
        T::matmul_t(tensor, &self.weight, out)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::f32::Tensor;

    #[test]
    fn test_linear() {
        let zeros = Tensor::zeros(vec![2, 2]);
        let weights = Tensor::zeros(vec![3, 2]);
        let bias = Tensor::zeros(vec![3]);
        let mut out = Tensor::zeros(vec![2, 3]);

        let linear = Linear::new(weights, bias);

        linear.forward(&zeros, &mut out).unwrap();
    }
}
