use super::ops;
use super::tensor::Tensor;
use crate::traits::{
    Tensor as TensorTrait, TensorAdd, TensorGelu, TensorMatmul, TensorMatmulT, TensorMul,
    TensorNormalize, TensorOps, TensorSelect, TensorSoftmax, TensorTanh,
};
use crate::SmeltError;

impl<'a> TensorTrait for Tensor {
    fn shape(&self) -> &[usize] {
        &self.shape()
    }
    fn zeros(shape: Vec<usize>) -> Self {
        // TODO
        Self::zeros(shape, 0).unwrap()
    }
}

impl<'a> TensorAdd<Tensor> for Tensor {
    fn add(x: &Self, y: &mut Self) -> Result<(), SmeltError> {
        ops::add(x, y)
    }
    fn broadcast_add(x: &Self, y: &mut Self) -> Result<(), SmeltError> {
        ops::broadcast_add(x, y)
    }
}

impl<'a> TensorMul<Tensor> for Tensor {
    fn mul(x: &Self, y: &mut Self) -> Result<(), SmeltError> {
        ops::mul(x, y)
    }
    fn broadcast_mul(x: &Self, y: &mut Self) -> Result<(), SmeltError> {
        ops::broadcast_mul(x, y)
    }
}

impl<'a> TensorNormalize<Tensor> for Tensor {
    fn normalize(x: &mut Self, epsilon: f32) -> Result<(), SmeltError> {
        ops::normalize(x, epsilon)
    }
}

impl<'a> TensorMatmul<Tensor> for Tensor {
    fn matmul(x: &Self, y: &Self, out: &mut Self) -> Result<(), SmeltError> {
        ops::matmul(x, y, out)
    }
}

impl<'a> TensorMatmulT<Tensor> for Tensor {
    fn matmul_t(x: &Self, y: &Self, out: &mut Self) -> Result<(), SmeltError> {
        ops::matmul_t(x, y, out)
    }
}

impl<'a> TensorSelect<Tensor> for Tensor {
    fn select(x: &[usize], weight: &Self, out: &mut Self) -> Result<(), SmeltError> {
        ops::select(x, weight, out)
    }
}

impl<'a> TensorGelu<Tensor> for Tensor {
    fn gelu(x: &mut Tensor) -> Result<(), SmeltError> {
        ops::gelu(x)?;
        Ok(())
    }
}

impl<'a> TensorTanh<Tensor> for Tensor {
    fn tanh(x: &mut Tensor) -> Result<(), SmeltError> {
        ops::tanh(x)?;
        Ok(())
    }
}

impl<'a> TensorSoftmax<Tensor> for Tensor {
    fn softmax(x: &mut Tensor) -> Result<(), SmeltError> {
        ops::softmax(x)
    }
}

impl<'a> TensorOps<Tensor> for Tensor {}
