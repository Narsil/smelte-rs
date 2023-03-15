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
    pub fn forward(&self, tensor: &T) -> Result<T, SmeltError> {
        let mut out = T::matmul_t(tensor, &self.weight)?;
        T::add(&self.bias, &mut out)?;
        Ok(out)
    }
}

/// LinearT layer, applies matmul(x, W) + b
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
    pub fn forward(&self, tensor: &T) -> Result<T, SmeltError> {
        let mut out = T::matmul(tensor, &self.weight)?;
        T::add(&self.bias, &mut out)?;
        Ok(out)
    }
}

/// UnbiasedLinear layer, applies matmul(x, W) + b
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
    pub fn forward(&self, tensor: &T) -> Result<T, SmeltError> {
        T::matmul(tensor, &self.weight)
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

        let linear = LinearT::new(weights, bias);

        linear.forward(&mut zeros).unwrap();
    }
}
