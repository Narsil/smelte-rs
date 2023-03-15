use super::ops;
use super::tensor::Tensor;
use crate::traits::{
    Tensor as TensorTrait, TensorAdd, TensorGelu, TensorMatmul, TensorMatmulT, TensorMul,
    TensorNormalize, TensorOps, TensorSelect, TensorSoftmax, TensorTanh,
};
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

impl<'a> TensorMul<Tensor<'a>> for Tensor<'a> {
    fn mul(x: &Self, y: &mut Self) -> Result<(), SmeltError> {
        ops::mul(x, y)
    }
}

impl<'a> TensorNormalize<Tensor<'a>> for Tensor<'a> {
    fn normalize(x: &mut Self, epsilon: f32) -> Result<(), SmeltError> {
        ops::normalize(x, epsilon)
    }
}

impl<'a> TensorMatmul<Tensor<'a>> for Tensor<'a> {
    fn matmul(x: &Self, y: &Self) -> Result<Tensor<'a>, SmeltError> {
        ops::matmul(x, y)
    }
}

impl<'a> TensorMatmulT<Tensor<'a>> for Tensor<'a> {
    fn matmul_t(x: &Self, y: &Self) -> Result<Tensor<'a>, SmeltError> {
        ops::matmul_t(x, y)
    }
}

impl<'a> TensorSelect<Tensor<'a>> for Tensor<'a> {
    fn select(x: &[usize], weight: &Tensor<'a>) -> Result<Tensor<'a>, SmeltError> {
        ops::select(x, weight)
    }
}

impl<'a> TensorGelu<Tensor<'a>> for Tensor<'a> {
    fn gelu(x: &mut Tensor<'a>) -> Result<(), SmeltError> {
        ops::apply(x, ops::gelu);
        Ok(())
    }
}

impl<'a> TensorTanh<Tensor<'a>> for Tensor<'a> {
    fn tanh(x: &mut Tensor<'a>) -> Result<(), SmeltError> {
        ops::apply(x, ops::inline_tanh);
        Ok(())
    }
}

impl<'a> TensorSoftmax<Tensor<'a>> for Tensor<'a> {
    fn softmax(x: &mut Tensor<'a>) -> Result<(), SmeltError> {
        ops::softmax(x)
    }
}

impl<'a> TensorOps<Tensor<'a>> for Tensor<'a> {}
