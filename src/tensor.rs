use std::borrow::Cow;

/// Error linked to the tensors themselves
#[derive(Debug)]
pub enum TensorError {
    /// The arguments to the tensor creation are invalid, the shape doesn't match
    /// the size of the buffer.
    InvalidBuffer {
        /// The size of the buffer sent
        buffer_size: usize,
        /// The shape of the tensor to create
        shape: Vec<usize>,
    },
}

/// Main tensor trait.
/// Tensors are meant to be CPU only for now.
pub trait Tensor {
    /// Give a pointer to the data buffer
    fn as_ptr(&self) -> *const f32 {
        self.data().as_ptr()
    }

    /// Give the shape of the tensor
    fn shape(&self) -> &[usize];

    /// Give a slice to the data buffer
    fn data(&self) -> &[f32];
}

/// Tensor trait for tensors that require to be modified
pub trait TensorMut: Tensor {
    /// Give a mutable slice to the data buffer
    fn data_mut(&mut self) -> &mut [f32];

    /// Give a mutable pointer to the data buffer
    fn as_mut_ptr(&mut self) -> *mut f32 {
        self.data_mut().as_mut_ptr()
    }

    /// Give a new empty mutable tensor
    fn zeros(shape: Vec<usize>) -> Self;
}

/// Readable only tensor. Mostly used for on disk tensors
/// which represent a given model
#[derive(Clone)]
pub struct ViewTensor<'data> {
    shape: Vec<usize>,
    data: Cow<'data, [f32]>,
}

impl<'data> Tensor for ViewTensor<'data> {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> &[f32] {
        self.data.as_ref()
    }
}
impl<'data> TensorMut for ViewTensor<'data> {
    fn data_mut(&mut self) -> &mut [f32] {
        self.data.to_mut()
    }

    fn zeros(shape: Vec<usize>) -> Self {
        let nelement: usize = shape.iter().product();
        let data = Cow::Owned(vec![0.0; nelement]);
        Self { shape, data }
    }
}

impl<'data> ViewTensor<'data> {
    /// Instantiate a new view tensor
    pub fn new(data: &'data [f32], shape: Vec<usize>) -> Result<Self, TensorError> {
        if data.len() != shape.iter().product::<usize>() {
            return Err(TensorError::InvalidBuffer {
                buffer_size: data.len(),
                shape,
            });
        }
        Ok(Self {
            shape,
            data: data.into(),
        })
    }
}

#[cfg(feature = "safetensors")]
mod safetensors {
    use super::ViewTensor;
    use safetensors::tensor::{Dtype, TensorView};
    use std::borrow::Cow;

    pub fn to_f32<'data>(view: &TensorView<'data>) -> Cow<'data, [f32]> {
        assert_eq!(view.dtype(), Dtype::F32);
        let v = view.data();
        if (v.as_ptr() as usize) % 4 == 0 {
            // SAFETY This is safe because we just checked that this
            // was correctly aligned.
            let data: &[f32] =
                unsafe { std::slice::from_raw_parts(v.as_ptr() as *const f32, v.len() / 4) };
            Cow::Borrowed(data)
        } else {
            let mut c = Vec::with_capacity(v.len() / 4);
            let mut i = 0;
            while i < v.len() {
                c.push(f32::from_le_bytes([v[i], v[i + 1], v[i + 2], v[i + 3]]));
                i += 4;
            }
            Cow::Owned(c)
        }
    }

    impl<'data> From<TensorView<'data>> for ViewTensor<'data> {
        fn from(view: TensorView<'data>) -> Self {
            let data = to_f32(&view);
            Self {
                data,
                shape: view.shape().to_vec(),
            }
        }
    }
}

/// Represent a new mutable tensor. Mostly useful during inference
/// for creating intermediary representations or input to a model
#[derive(Debug, Clone)]
pub struct OwnedTensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl Tensor for OwnedTensor {
    fn as_ptr(&self) -> *const f32 {
        self.data.as_ptr()
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> &[f32] {
        &self.data
    }
}

impl TensorMut for OwnedTensor {
    fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }
    fn zeros(shape: Vec<usize>) -> Self {
        let nelement: usize = shape.iter().product();
        let data = vec![0.0; nelement];
        Self { shape, data }
    }
}
impl OwnedTensor {
    /// Create a new OwnedTensor
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, TensorError> {
        if data.len() != shape.iter().product::<usize>() {
            return Err(TensorError::InvalidBuffer {
                buffer_size: data.len(),
                shape,
            });
        }
        Ok(Self { shape, data })
    }
}
