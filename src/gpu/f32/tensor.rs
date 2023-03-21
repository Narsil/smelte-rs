use crate::SmeltError;
use cudarc::driver::{CudaDevice, CudaSlice, DriverError};
use std::sync::Arc;

/// Tensor, can own, or borrow the underlying tensor
#[derive(Clone)]
pub struct Tensor {
    shape: Vec<usize>,
    device: Arc<CudaDevice>,
    device_id: usize,
    data: CudaSlice<f32>,
}

impl Tensor {
    /// The shape of the tensor
    /// ```
    /// use smelte_rs::gpu::f32::Tensor;
    ///
    /// let tensor = Tensor::zeros(vec![2, 2], 0).unwrap();
    /// assert_eq!(tensor.shape(), vec![2, 2]);
    /// ```
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The [CudaSlice] holding the data
    pub fn data(&self) -> &CudaSlice<f32> {
        &self.data
    }

    /// A mutable borrow of [CudaSlice] holding the data
    pub fn data_mut(&mut self) -> &mut CudaSlice<f32> {
        &mut self.data
    }

    /// The device of the device
    pub fn device(&self) -> Arc<CudaDevice> {
        self.device.clone()
    }

    /// The device id
    pub fn device_id(&self) -> usize {
        self.device_id
    }

    /// Creates a new nulled tensor with given shape
    /// ```
    /// use smelte_rs::gpu::f32::Tensor;
    ///
    /// let tensor = Tensor::zeros(vec![2, 2], 0).unwrap();
    /// ```
    pub fn zeros(shape: Vec<usize>, device_id: usize) -> Result<Self, DriverError> {
        let nelement: usize = shape.iter().product();
        let device = CudaDevice::new(device_id)?;
        let data: CudaSlice<f32> = device.alloc_zeros(nelement)?;
        Ok(Self {
            shape,
            data,
            device,
            device_id,
        })
    }

    /// Creates a tensor from a cpu [Vec].
    pub fn from_cpu(data: &[f32], shape: Vec<usize>, device_id: usize) -> Result<Self, SmeltError> {
        let device = CudaDevice::new(device_id)?;
        if data.len() != shape.iter().product::<usize>() {
            return Err(SmeltError::InvalidBuffer {
                buffer_size: data.len(),
                shape,
            });
        }
        let data = device.htod_sync_copy(data)?;
        Ok(Self {
            device,
            device_id,
            data,
            shape,
        })
    }

    /// Returns a cpu vec containing copied data from the device.
    pub fn cpu_data(&self) -> Result<Vec<f32>, SmeltError> {
        let cpu_data = self.device.dtoh_sync_copy(&self.data)?;
        Ok(cpu_data)
    }
}
