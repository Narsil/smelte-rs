use crate::TensorError;
use half::f16;
use std::borrow::Cow;

/// Tensor, can own, or borrow the underlying tensor
#[derive(Clone)]
pub struct Tensor<'data> {
    shape: Vec<usize>,
    data: Cow<'data, [f16]>,
}

impl<'data> Tensor<'data> {
    /// The shape of the tensor
    /// ```
    /// use smelte-rs::cpu::f16::Tensor;
    ///
    /// let tensor = Tensor::zeros(vec![2, 2]);
    /// assert_eq!(tensor.shape(), vec![2, 2]);
    /// ```
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// A slice to the underlying tensor data
    /// ```
    /// use smelte-rs::cpu::f16::Tensor;
    ///
    /// let tensor = Tensor::zeros(vec![2, 2]);
    /// assert_eq!(tensor.data(), vec![0.0; 4]);
    /// ```
    pub fn data(&self) -> &[f16] {
        self.data.as_ref()
    }

    /// A mutable slice to the underlying tensor data
    /// ```
    /// use smelte-rs::cpu::f16::Tensor;
    ///
    /// let mut tensor = Tensor::zeros(vec![2, 2]);
    /// tensor.data_mut().iter_mut().for_each(|v| *v += 1.0);
    /// assert_eq!(tensor.data(), vec![1.0; 4]);
    /// ```
    pub fn data_mut(&mut self) -> &mut [f16] {
        self.data.to_mut()
    }

    /// Creates a new nulled tensor with given shape
    /// ```
    /// use smelte-rs::cpu::f16::Tensor;
    ///
    /// let tensor = Tensor::zeros(vec![2, 2]);
    /// ```
    pub fn zeros(shape: Vec<usize>) -> Self {
        let nelement: usize = shape.iter().product();
        let data = Cow::Owned(vec![f16::from_f32(0.0); nelement]);
        Self { shape, data }
    }

    /// Creates a new borrowed tensor with given shape. Can fail if data doesn't match the shape
    /// ```
    /// use smelte-rs::cpu::f16::Tensor;
    ///
    /// let data = [1.0, 2.0, 3.0, 4.0];
    /// let tensor = Tensor::borrowed(&data, vec![2, 2]).unwrap();
    /// ```
    pub fn borrowed(data: &'data [f16], shape: Vec<usize>) -> Result<Self, TensorError> {
        let cow: Cow<'data, [f16]> = data.into();
        Self::new(cow, shape)
    }

    /// Creates a new tensor with given shape. Can fail if data doesn't match the shape
    /// ```
    /// use smelte-rs::cpu::f16::Tensor;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let tensor = Tensor::new(data, vec![2, 2]).unwrap();
    /// ```
    pub fn new<T>(data: T, shape: Vec<usize>) -> Result<Self, TensorError>
    where
        T: Into<Cow<'data, [f16]>>,
    {
        let data = data.into();
        if data.len() != shape.iter().product::<usize>() {
            return Err(TensorError::InvalidBuffer {
                buffer_size: data.len(),
                shape,
            });
        }
        Ok(Self { shape, data })
    }
}
