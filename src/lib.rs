#![deny(missing_docs)]
//! Placeholder

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

/// The various CPU implementations
pub mod cpu;

#[cfg(test)]
mod tests {
    pub(crate) fn simplify(data: &[f32]) -> Vec<f32> {
        let precision = 3;
        let m = 10.0 * 10.0f32.powf(precision as f32);
        data.iter().map(|x| (x * m).round() / m).collect()
    }

    // fn assert_float_eq(left: &[f32], right: &[f32]) {
    //     assert_eq!(left.len(), right.len());

    //     left.iter().zip(right.iter()).for_each(|(l, r)| {
    //         assert!(
    //             (l - r).abs() / l.abs() < 1e-4,
    //             "{l} != {r}\n{left:?}\n{right:?}"
    //         );
    //     });
    // }
}
