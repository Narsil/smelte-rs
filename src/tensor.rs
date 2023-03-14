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

/// Readable only tensor. Mostly used for on disk tensors
/// which represent a given model
#[derive(Clone)]
pub struct Tensor<'data> {
    shape: Vec<usize>,
    data: Cow<'data, [f32]>,
}

impl<'data> Tensor<'data> {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[f32] {
        self.data.as_ref()
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        self.data.to_mut()
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let nelement: usize = shape.iter().product();
        let data = Cow::Owned(vec![0.0; nelement]);
        Self { shape, data }
    }
    pub fn borrowed(data: &'data [f32], shape: Vec<usize>) -> Result<Self, TensorError> {
        let cow: Cow<'data, [f32]> = data.into();
        Self::new(cow, shape)
    }

    pub fn new<T>(data: T, shape: Vec<usize>) -> Result<Self, TensorError>
    where
        T: Into<Cow<'data, [f32]>>,
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

// #[cfg(feature = "safetensors")]
// mod safetensors {
//     use super::ViewTensor;
//     use safetensors::tensor::{Dtype, TensorView};
//     use std::borrow::Cow;
//
//     pub fn to_f32<'data>(view: &TensorView<'data>) -> Cow<'data, [f32]> {
//         assert_eq!(view.dtype(), Dtype::F32);
//         let v = view.data();
//         if (v.as_ptr() as usize) % 4 == 0 {
//             // SAFETY This is safe because we just checked that this
//             // was correctly aligned.
//             let data: &[f32] =
//                 unsafe { std::slice::from_raw_parts(v.as_ptr() as *const f32, v.len() / 4) };
//             Cow::Borrowed(data)
//         } else {
//             let mut c = Vec::with_capacity(v.len() / 4);
//             let mut i = 0;
//             while i < v.len() {
//                 c.push(f32::from_le_bytes([v[i], v[i + 1], v[i + 2], v[i + 3]]));
//                 i += 4;
//             }
//             Cow::Owned(c)
//         }
//     }
//
//     impl<'data> From<TensorView<'data>> for ViewTensor<'data> {
//         fn from(view: TensorView<'data>) -> Self {
//             let data = to_f32(&view);
//             Self {
//                 data,
//                 shape: view.shape().to_vec(),
//             }
//         }
//     }
// }
