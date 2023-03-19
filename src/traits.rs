use crate::SmeltError;

/// TODO
pub trait Tensor: Clone {
    /// TODO
    fn shape(&self) -> &[usize];
    /// TODO
    fn zeros(shape: Vec<usize>) -> Self;
}

/// All common tensor operations
pub trait TensorOps<T>:
    TensorMatmul<T>
    + TensorMatmulT<T>
    + TensorAdd<T>
    + TensorMul<T>
    + TensorNormalize<T>
    + TensorSelect<T>
    + TensorGelu<T>
    + TensorTanh<T>
    + TensorSoftmax<T>
{
}

/// TODO
pub trait TensorMatmul<T> {
    /// TODO
    fn matmul(a: &T, b: &T, c: &mut T) -> Result<(), SmeltError>;
}

/// TODO
pub trait TensorMatmulT<T> {
    /// TODO
    fn matmul_t(a: &T, b: &T, c: &mut T) -> Result<(), SmeltError>;
}

/// TODO
pub trait TensorAdd<T> {
    /// TODO
    fn add(a: &T, b: &mut T) -> Result<(), SmeltError>;
    /// TODO
    fn broadcast_add(a: &T, b: &mut T) -> Result<(), SmeltError>;
}

/// TODO
pub trait TensorMul<T> {
    /// TODO
    fn mul(a: &T, b: &mut T) -> Result<(), SmeltError>;
    /// TODO
    fn broadcast_mul(a: &T, b: &mut T) -> Result<(), SmeltError>;
}

/// TODO
pub trait TensorNormalize<T> {
    /// TODO
    fn normalize(x: &mut T, epsilon: f32) -> Result<(), SmeltError>;
}

/// TODO
pub trait TensorSelect<T> {
    /// TODO
    fn select(x: &[usize], weight: &T, out: &mut T) -> Result<(), SmeltError>;
}

/// TODO
pub trait TensorGelu<T> {
    /// TODO
    fn gelu(x: &mut T) -> Result<(), SmeltError>;
}

/// TODO
pub trait TensorTanh<T> {
    /// TODO
    fn tanh(x: &mut T) -> Result<(), SmeltError>;
}

/// TODO
pub trait TensorSoftmax<T> {
    /// TODO
    fn softmax(x: &mut T) -> Result<(), SmeltError>;
}
