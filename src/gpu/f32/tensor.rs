use crate::SmeltError;
use cudarc::cublas::safe::CudaBlas;
use cudarc::driver::{CudaDevice, CudaSlice, DriverError};
use std::sync::Arc;

/// Tensor, can own, or borrow the underlying tensor
#[derive(Clone)]
pub struct Tensor {
    shape: Vec<usize>,
    device: Device,
    data: CudaSlice<f32>,
}

/// The GPU device, contains its id, a cuda handle and a cublas handle
#[derive(Clone)]
pub struct Device {
    device: Arc<CudaDevice>,
    device_id: usize,
    blas: Arc<CudaBlas>,
}

impl Device {
    /// TODO
    pub fn new(device_id: usize) -> Result<Self, SmeltError> {
        let device = CudaDevice::new(device_id)?;
        let blas = Arc::new(CudaBlas::new(device.clone())?);
        Ok(Self {
            device,
            device_id,
            blas,
        })
    }
}

impl Tensor {
    /// The shape of the tensor
    /// ```
    /// use smelte_rs::gpu::f32::{Tensor, Device};
    ///
    /// let device = Device::new(0).unwrap();
    /// let tensor = Tensor::zeros(vec![2, 2], &device).unwrap();
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
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The device of the device
    pub fn cuda(&self) -> Arc<CudaDevice> {
        self.device.device.clone()
    }

    /// The CudaBlas handle
    pub fn blas(&self) -> Arc<CudaBlas> {
        self.device.blas.clone()
    }

    /// The device id
    pub fn device_id(&self) -> usize {
        self.device.device_id
    }

    /// Creates a new nulled tensor with given shape
    /// ```
    /// use smelte_rs::gpu::f32::{Tensor, Device};
    ///
    /// let device = Device::new(0).unwrap();
    /// let tensor = Tensor::zeros(vec![2, 2], &device).unwrap();
    /// ```
    pub fn zeros(shape: Vec<usize>, device: &Device) -> Result<Self, DriverError> {
        let nelement: usize = shape.iter().product();
        let data: CudaSlice<f32> = device.device.alloc_zeros(nelement)?;
        Ok(Self {
            shape,
            data,
            device: device.clone(),
        })
    }

    /// Creates a tensor from a cpu [Vec].
    pub fn from_cpu(data: &[f32], shape: Vec<usize>, device: &Device) -> Result<Self, SmeltError> {
        if data.len() != shape.iter().product::<usize>() {
            return Err(SmeltError::InvalidBuffer {
                buffer_size: data.len(),
                shape,
            });
        }
        let data = device.device.htod_sync_copy(data).unwrap();
        Ok(Self {
            device: device.clone(),
            data,
            shape,
        })
    }

    /// Returns a cpu vec containing copied data from the device.
    pub fn cpu_data(&self) -> Result<Vec<f32>, SmeltError> {
        let cpu_data = self.device.device.dtoh_sync_copy(&self.data)?;
        Ok(cpu_data)
    }
}
