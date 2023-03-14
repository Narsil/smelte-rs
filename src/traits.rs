use crate::SmeltError;

/// TODO
pub trait Tensor {
    /// TODO
    fn shape(&self) -> &[usize];
    /// TODO
    fn zeros(shape: Vec<usize>) -> Self;
}

/// TODO
pub trait TensorMatmul<T> {
    /// TODO
    fn matmul_t(a: &T, b: &T, c: &mut T) -> Result<(), SmeltError>;
}

/// TODO
pub trait TensorAdd<T> {
    /// TODO
    fn add(a: &T, b: &mut T) -> Result<(), SmeltError>;
}
