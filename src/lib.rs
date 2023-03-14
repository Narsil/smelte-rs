#![deny(missing_docs)]
//! Placeholder

/// The various CPU implementations
pub mod cpu;

/// The neural networks
pub mod nn;

/// The traits for generic implementations
pub mod traits;

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

/// Potential errors when using the library
#[derive(Debug)]
pub enum SmeltError {
    /// The operation could not succeed because the shapes are not valid.
    DimensionMismatch {
        /// The shape that we should have seen
        expected: Vec<usize>,
        /// The shape that we received
        got: Vec<usize>,
    },
    /// The tensor given has insufficient rank (rank 2 means a tensor that has a shape of length 2)
    InsufficientRank {
        /// The minimum rank that we expect
        minimum_rank: usize,
    },
    /// The tensor given has not the expected rank (rank 2 means a tensor that has a shape of length 2)
    InvalidRank {
        /// The rank that we expect
        expected_rank: usize,
    },
    /// The tensor given has not enough room for the operations
    VectorTooSmall {
        /// The minimum size that we expect
        minimum: usize,
    },
}

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
