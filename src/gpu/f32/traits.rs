use super::ops;
use super::tensor::{Device, Tensor};
use crate::traits::{
    Device as DeviceTrait, Tensor as TensorTrait, TensorAdd, TensorGelu, TensorMatmul,
    TensorMatmulT, TensorMul, TensorNormalize, TensorOps, TensorSelect, TensorSoftmax, TensorTanh,
};
use crate::SmeltError;

impl TensorTrait for Tensor {
    type Device = Device;

    fn shape(&self) -> &[usize] {
        &self.shape()
    }

    fn device(&self) -> &Device {
        &self.device()
    }
}

impl DeviceTrait for Device {
    type Tensor = Tensor;
    fn zeros(&self, shape: Vec<usize>) -> Result<Self::Tensor, SmeltError> {
        Ok(Self::Tensor::zeros(shape, self)?)
    }
}

impl TensorAdd<Tensor> for Tensor {
    fn add(x: &Self, y: &mut Self) -> Result<(), SmeltError> {
        ops::add(x, y)
    }
    fn broadcast_add(x: &Self, y: &mut Self) -> Result<(), SmeltError> {
        ops::broadcast_add(x, y)
    }
}

impl TensorMul<Tensor> for Tensor {
    fn mul(x: &Self, y: &mut Self) -> Result<(), SmeltError> {
        ops::mul(x, y)
    }
    fn broadcast_mul(x: &Self, y: &mut Self) -> Result<(), SmeltError> {
        ops::broadcast_mul(x, y)
    }
}

impl TensorNormalize<Tensor> for Tensor {
    fn normalize(x: &mut Self, epsilon: f32) -> Result<(), SmeltError> {
        ops::normalize(x, epsilon)
    }
}

impl TensorMatmul<Tensor> for Tensor {
    fn matmul(x: &Self, y: &Self, out: &mut Self) -> Result<(), SmeltError> {
        ops::matmul(x, y, out)
    }
}

impl TensorMatmulT<Tensor> for Tensor {
    fn matmul_t(x: &Self, y: &Self, out: &mut Self) -> Result<(), SmeltError> {
        ops::matmul_t(x, y, out)
    }
}

impl TensorSelect<Tensor> for Tensor {
    fn select(x: &[usize], weight: &Self, out: &mut Self) -> Result<(), SmeltError> {
        ops::select(x, weight, out)
    }
}

impl TensorGelu<Tensor> for Tensor {
    fn gelu(x: &mut Tensor) -> Result<(), SmeltError> {
        ops::gelu(x)?;
        Ok(())
    }
}

impl TensorTanh<Tensor> for Tensor {
    fn tanh(x: &mut Tensor) -> Result<(), SmeltError> {
        ops::tanh(x)?;
        Ok(())
    }
}

impl TensorSoftmax<Tensor> for Tensor {
    fn softmax(x: &mut Tensor) -> Result<(), SmeltError> {
        ops::softmax(x)
    }
}

impl TensorOps<Tensor> for Tensor {}
