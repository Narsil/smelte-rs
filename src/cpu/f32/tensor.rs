use crate::SmeltError;
use std::borrow::Cow;

/// Tensor, can own, or borrow the underlying tensor
#[derive(Clone)]
pub struct Tensor {
    pub(super) shape: Vec<usize>,
    data: Cow<'static, [f32]>,
}

/// The CPU device
#[derive(Copy, Clone)]
pub struct Device {}

impl Tensor {
    /// The shape of the tensor
    /// ```
    /// use smelte_rs::cpu::f32::Tensor;
    ///
    /// let tensor = Tensor::zeros(vec![2, 2]);
    /// assert_eq!(tensor.shape(), vec![2, 2]);
    /// ```
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// A slice to the underlying tensor data
    /// ```
    /// use smelte_rs::cpu::f32::Tensor;
    ///
    /// let tensor = Tensor::zeros(vec![2, 2]);
    /// assert_eq!(tensor.data(), vec![0.0; 4]);
    /// ```
    pub fn data(&self) -> &[f32] {
        self.data.as_ref()
    }

    /// A slice to the underlying tensor data.
    /// Exists uniquely for symetry with gpu Tensor.
    /// ```
    /// use smelte_rs::cpu::f32::Tensor;
    ///
    /// let tensor = Tensor::zeros(vec![2, 2]);
    /// assert_eq!(tensor.cpu_data().unwrap(), vec![0.0; 4]);
    /// ```
    pub fn cpu_data(&self) -> Result<&[f32], SmeltError> {
        Ok(self.data.as_ref())
    }

    /// A mutable slice to the underlying tensor data
    /// ```
    /// use smelte_rs::cpu::f32::Tensor;
    ///
    /// let mut tensor = Tensor::zeros(vec![2, 2]);
    /// tensor.data_mut().iter_mut().for_each(|v| *v += 1.0);
    /// assert_eq!(tensor.data(), vec![1.0; 4]);
    /// ```
    pub fn data_mut(&mut self) -> &mut [f32] {
        self.data.to_mut()
    }

    /// Creates a new nulled tensor with given shape
    /// ```
    /// use smelte_rs::cpu::f32::Tensor;
    ///
    /// let tensor = Tensor::zeros(vec![2, 2]);
    /// ```
    pub fn zeros(shape: Vec<usize>) -> Self {
        let nelement: usize = shape.iter().product();
        let data = Cow::Owned(vec![0.0; nelement]);
        Self { shape, data }
    }

    /// Creates a new borrowed tensor with given shape. Can fail if data doesn't match the shape
    /// ```
    /// use smelte_rs::cpu::f32::Tensor;
    ///
    /// let tensor = Tensor::borrowed(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// ```
    pub fn borrowed(data: &'static [f32], shape: Vec<usize>) -> Result<Self, SmeltError> {
        let cow: Cow<'static, [f32]> = data.into();
        Self::new(cow, shape)
    }

    /// Creates a new tensor with given shape. Can fail if data doesn't match the shape
    /// ```
    /// use smelte_rs::cpu::f32::Tensor;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let tensor = Tensor::new(data, vec![2, 2]).unwrap();
    /// ```
    pub fn new<T>(data: T, shape: Vec<usize>) -> Result<Self, SmeltError>
    where
        T: Into<Cow<'static, [f32]>>,
    {
        let data = data.into();
        if data.len() != shape.iter().product::<usize>() {
            return Err(SmeltError::InvalidBuffer {
                buffer_size: data.len(),
                shape,
            });
        }
        Ok(Self { shape, data })
    }

    /// Creates a new tensor with given shape. Can fail if data doesn't match the shape
    /// Exists only for symetry with gpu tensor.
    /// ```
    /// use smelte_rs::cpu::f32::{Tensor, Device};
    ///
    /// let device = Device{};
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let tensor = Tensor::from_cpu(data, vec![2, 2], &device).unwrap();
    /// ```
    pub fn from_cpu<T>(data: T, shape: Vec<usize>, _device: &Device) -> Result<Self, SmeltError>
    where
        T: Into<Cow<'static, [f32]>>,
    {
        let data = data.into();
        if data.len() != shape.iter().product::<usize>() {
            return Err(SmeltError::InvalidBuffer {
                buffer_size: data.len(),
                shape,
            });
        }
        Ok(Self { shape, data })
    }
}
