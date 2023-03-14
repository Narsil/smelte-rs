use super::ops;
use super::tensor::Tensor;
use crate::traits::{Tensor as TensorTrait, TensorAdd, TensorMatmul};
use crate::SmeltError;

impl<'a> TensorTrait for Tensor<'a> {
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn zeros(shape: Vec<usize>) -> Self {
        Self::zeros(shape)
    }
}

impl<'a> TensorAdd<Tensor<'a>> for Tensor<'a> {
    fn add(x: &Self, y: &mut Self) -> Result<(), SmeltError> {
        ops::add(x, y)
    }
}

impl<'a> TensorMatmul<Tensor<'a>> for Tensor<'a> {
    fn matmul_t(x: &Self, y: &Self, z: &mut Self) -> Result<(), SmeltError> {
        ops::matmul_t(x, y, z)
    }
}
