use crate::tensor::{Tensor, TensorMut};
use rayon::prelude::*;

#[cfg(feature = "cblas")]
use cblas_sys::{
    cblas_sgemm as sgemm, CblasColMajor as ColMajor, CblasNoTrans as NoTr,
    CblasRowMajor as RowMajor, CblasTrans as Tr,
};
#[cfg(feature = "intel-mkl")]
use mkl_sys::{
    cblas_sgemm_batch_strided as sgemm_batch_strided, CBLAS_LAYOUT_CblasColMajor as ColMajor,
    CBLAS_LAYOUT_CblasRowMajor as RowMajor, CBLAS_TRANSPOSE_CblasNoTrans as NoTr,
    CBLAS_TRANSPOSE_CblasTrans as Tr,
};

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

/// Operation for selecting entire rows within tensor `weights`. Each `id` is the index
/// of the row.
pub fn select<T: Tensor, TM: TensorMut>(
    ids: &[u32],
    weights: &T,
    out: &mut TM,
) -> Result<(), SmeltError> {
    let hidden_dim = weights.shape()[1];
    let sequence_length = ids.len();
    if out.shape() != [sequence_length, hidden_dim] {
        return Err(SmeltError::DimensionMismatch {
            expected: vec![sequence_length, hidden_dim],
            got: out.shape().to_vec(),
        });
    }
    for (i, id) in ids.iter().enumerate() {
        let id = *id as usize;
        let weight_offset = id * hidden_dim;
        let data_offset = i * hidden_dim;
        out.data_mut()[data_offset..data_offset + hidden_dim]
            .copy_from_slice(&weights.data()[weight_offset..weight_offset + hidden_dim]);
    }
    Ok(())
}

/// Regular matrix multiplication
pub fn matmul<A: Tensor, B: Tensor, TM: TensorMut>(
    a: &A,
    b: &B,
    c: &mut TM,
) -> Result<(), SmeltError> {
    g_matmul::<false, A, B, TM>(a, b, c)
}

/// Matrix multiplication matmul(A, B.transposed())
pub fn matmul_t<A: Tensor, B: Tensor, TM: TensorMut>(
    a: &A,
    b: &B,
    c: &mut TM,
) -> Result<(), SmeltError> {
    g_matmul::<true, A, B, TM>(a, b, c)
}

#[inline]
fn g_matmul<const TRANSPOSE: bool, A: Tensor, B: Tensor, TM: TensorMut>(
    a: &A,
    b: &B,
    c: &mut TM,
) -> Result<(), SmeltError> {
    let dim = a.shape().len();

    if dim < 2 {
        return Err(SmeltError::InsufficientRank { minimum_rank: 2 });
    }
    if b.shape().len() != dim {
        return Err(SmeltError::InvalidRank { expected_rank: dim });
    }
    if c.shape().len() != dim {
        return Err(SmeltError::InvalidRank { expected_rank: dim });
    }

    let m = a.shape()[dim - 2];
    let k = a.shape()[dim - 1];

    let mut expected_c = a.shape().to_vec();
    let mut expected_b = a.shape().to_vec();

    let (expected_b, n) = if TRANSPOSE {
        let n = b.shape()[dim - 2];
        expected_b[dim - 2] = n;
        expected_b[dim - 1] = k;
        (expected_b, n)
    } else {
        let n = b.shape()[dim - 1];
        expected_b[dim - 2] = k;
        expected_b[dim - 1] = n;
        (expected_b, n)
    };

    expected_c[dim - 2] = m;
    expected_c[dim - 1] = n;

    if expected_b != b.shape() {
        return Err(SmeltError::DimensionMismatch {
            expected: expected_b,
            got: b.shape().to_vec(),
        });
    }

    if expected_c != c.shape() {
        return Err(SmeltError::DimensionMismatch {
            expected: expected_c,
            got: c.shape().to_vec(),
        });
    }

    let batching: usize = a.shape()[..dim - 2].iter().product();
    let a_skip: usize = m * k;
    let b_skip: usize = n * k;
    let c_skip: usize = m * n;

    let ar = k as isize;
    let ac = 1;
    let (br, bc) = if TRANSPOSE {
        (1, b.shape()[dim - 1] as isize)
    } else {
        (b.shape()[dim - 1] as isize, 1)
    };
    let cr = n as isize;
    let cc = 1;

    // (0..batching).for_each(|step| {
    let ap = &a.data();
    let bp = &b.data();
    let cp = &mut c.data_mut();

    unsafe {
        let (m, n, k) = (m as libc::c_int, n as libc::c_int, k as libc::c_int);
        let (layout, a_tr, b_tr, lda, ldb, ldc) = if cr < cc {
            let (lda, a_tr) = if ar < ac { (m, NoTr) } else { (k, Tr) };
            let (ldb, b_tr) = if br < bc { (k, NoTr) } else { (n, Tr) };
            (ColMajor, a_tr, b_tr, lda, ldb, m)
        } else {
            let (lda, a_tr) = if ar < ac { (m, Tr) } else { (k, NoTr) };
            let (ldb, b_tr) = if br < bc { (k, Tr) } else { (n, NoTr) };
            (RowMajor, a_tr, b_tr, lda, ldb, n)
        };
        sgemm_batch_strided(
            layout,
            a_tr,
            b_tr,
            m,
            n,
            k,
            1.0,
            ap.as_ptr(),
            lda,
            a_skip as i32,
            bp.as_ptr(),
            ldb,
            b_skip as i32,
            1.0,
            cp.as_mut_ptr(),
            ldc,
            c_skip as i32,
            batching as i32,
        )
    }
    // });
    Ok(())
}

/// tensor elementwise addition. b += a.
/// a is automatically broadcasted.
pub fn add<T: Tensor, TM: TensorMut>(a: &T, b: &mut TM) -> Result<(), SmeltError> {
    if a.shape() == b.shape() {
        a.data()
            .iter()
            .zip(b.data_mut().iter_mut())
            .for_each(|(left, right)| *right += left);
        Ok(())
    } else if &b.shape()[1..] == a.shape() {
        let n = b.shape()[0];
        (0..n).for_each(|i| {
            a.data()
                .iter()
                .zip(b.data_mut().iter_mut().skip(i * a.shape()[0]))
                .for_each(|(left, right)| *right += left);
        });
        Ok(())
    } else {
        Err(SmeltError::DimensionMismatch {
            expected: b.shape().to_vec(),
            got: a.shape().to_vec(),
        })
    }
}

/// tensor elementwise multiplication. b *= a.
/// a is automatically broadcasted.
pub fn mul<T: Tensor, TM: TensorMut>(a: &T, b: &mut TM) -> Result<(), SmeltError> {
    if a.shape() == b.shape() {
        a.data()
            .iter()
            .zip(b.data_mut().iter_mut())
            .for_each(|(left, right)| *right *= left);
        Ok(())
    } else if &b.shape()[1..] == a.shape() {
        let n = b.shape()[0];
        (0..n).for_each(|i| {
            a.data()
                .iter()
                .zip(b.data_mut().iter_mut().skip(i * a.shape()[0]))
                .for_each(|(left, right)| *right *= left);
        });
        Ok(())
    } else {
        Err(SmeltError::DimensionMismatch {
            expected: b.shape().to_vec(),
            got: a.shape().to_vec(),
        })
    }
}

/// Basic operation for the layernorm.
/// x = (x - x.mean()) / (x.var() + epsilon)
/// `mean` and `var` do not have to be initialized, they are simply passed to
/// avoid allocation.
pub fn normalize<TM: TensorMut>(x: &mut TM, epsilon: f32) -> Result<(), SmeltError> {
    let dim = x.shape().len();
    let size = x.shape()[dim - 1];
    x.data_mut().par_chunks_mut(size).for_each(|chunk| {
        let sum: f32 = chunk.iter().sum();
        let mean = sum / size as f32;
        chunk.iter_mut().for_each(|v| *v -= mean);
        let var: f32 = chunk.iter().map(|v| v.powf(2.0)).sum();
        let var = var / size as f32;
        let stddev: f32 = (var + epsilon).sqrt();
        chunk.iter_mut().for_each(|v| *v /= stddev);
    });
    Ok(())
}

#[inline]
fn g_softmax<const CAUSAL: bool, TM: TensorMut>(
    x: &mut TM,
    max: &mut [f32],
    past_sequence_length: usize,
) -> Result<(), SmeltError> {
    let dim = x.shape().len();

    let m = x.shape()[dim - 2];
    let n = x.shape()[dim - 1];
    let b: usize = x.shape()[..dim - 2].iter().product();
    if max.len() < b * m {
        return Err(SmeltError::VectorTooSmall { minimum: b * m });
    }

    x.data_mut()
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(i, chunk)| {
            let i = i % m;
            let mut current_max = f32::NEG_INFINITY;
            for (j, &v) in chunk.iter().enumerate() {
                if (!CAUSAL || i + past_sequence_length >= j) && v > current_max {
                    current_max = v;
                }
            }
            for v in chunk.iter_mut() {
                *v -= current_max;
                *v = (*v).exp();
            }
            let mut sum = 0.0;
            for (j, &v) in chunk.iter().enumerate() {
                if !CAUSAL || i + past_sequence_length >= j {
                    sum += v;
                }
            }
            for (j, v) in chunk.iter_mut().enumerate() {
                if !CAUSAL || i + past_sequence_length >= j {
                    *v /= sum;
                } else {
                    *v = 0.0;
                }
            }
        });
    Ok(())
}

/// Softmax on the last dimension for tensor `x`
pub fn softmax<TM: TensorMut>(x: &mut TM, max: &mut [f32]) -> Result<(), SmeltError> {
    g_softmax::<false, TM>(x, max, 0)
}

/// Causal softmax on the last dimension for tensor `x`. The causality is determined by the
/// shape of `x` and `past_sequence_length` which defines how big is the missing part of the
/// square.
pub fn causal_softmax<TM: TensorMut>(
    x: &mut TM,
    max: &mut [f32],
    past_sequence_length: usize,
) -> Result<(), SmeltError> {
    g_softmax::<true, TM>(x, max, past_sequence_length)
}

/// Argmax of the last dimension of tensor `x `.
pub fn special_argmax<T: Tensor>(x: &T) -> Result<usize, SmeltError> {
    if x.shape().len() != 2 {
        return Err(SmeltError::InvalidRank { expected_rank: 2 });
    }
    let n = x.shape()[0];
    let m = x.shape()[1];

    let mut max = f32::NEG_INFINITY;
    let mut max_id = usize::MAX;
    for (i, &v) in x.data().iter().skip((n - 1) * m).enumerate() {
        if v > max {
            max = v;
            max_id = i;
        }
    }
    Ok(max_id)
}

/// utility function to use a faster but less precise tanh
pub fn faster_tanh(x: f32) -> f32 {
    let x2 = x * x;
    let x3 = x2 * x;
    let x5 = x3 * x2;

    let a = x + (0.16489087 * x3) + (0.00985468 * x5);

    a / (1.0 + (a * a)).sqrt()
}

/// `gelu` operation
/// <https://en.wikipedia.org/wiki/Activation_function#Comparison_of_activation_functions>
/// but using [faster_tanh]
pub fn faster_gelu(v: f32) -> f32 {
    0.5 * (v)
        * (1.0 + faster_tanh((2.0f32 / std::f32::consts::PI).sqrt() * (v + 0.044715 * v.powf(3.0))))
}

/// `gelu` operation
/// <https://en.wikipedia.org/wiki/Activation_function#Comparison_of_activation_functions>
pub fn gelu(v: f32) -> f32 {
    0.5 * (v)
        * (1.0 + f32::tanh((2.0f32 / std::f32::consts::PI).sqrt() * v * (1.0 + 0.044715 * v * v)))
}

/// Applies `func` to every item of the tensor
pub fn par_apply<T: TensorMut, F: Fn(f32) -> f32 + Sync>(x: &mut T, func: F) {
    x.data_mut().par_iter_mut().for_each(|v| *v = func(*v));
}

/// Applies `func` to every item of the tensor
pub fn apply<T: TensorMut, F: Fn(f32) -> f32 + Sync>(x: &mut T, func: F) {
    x.data_mut().iter_mut().for_each(|v| *v = func(*v));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{OwnedTensor, ViewTensor};
    use crate::tests::simplify;

    #[test]
    fn simple_matmul() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let a = OwnedTensor::new(data, vec![2, 2]).unwrap();
        let data = [1.0, 2.0, 3.0, 4.0];
        let b = ViewTensor::new(&data, vec![2, 2]).unwrap();
        let data = vec![0.0; 4];
        let mut c = OwnedTensor::new(data, vec![2, 2]).unwrap();

        matmul(&a, &b, &mut c).unwrap();
        assert_eq!(c.data(), &[7.0, 10.0, 15.0, 22.0]);

        let data = vec![1.0, 2.0];
        let a = OwnedTensor::new(data, vec![2, 1]).unwrap();
        let data = [3.0, 4.0];
        let b = ViewTensor::new(&data, vec![1, 2]).unwrap();
        let data = vec![0.0; 4];
        let mut c = OwnedTensor::new(data, vec![2, 2]).unwrap();
        matmul(&a, &b, &mut c).unwrap();
        assert_eq!(c.data(), &[3.0, 4.0, 6.0, 8.0]);

        let data: Vec<_> = (0..6).map(|i| i as f32).collect();
        let a = OwnedTensor::new(data, vec![2, 3]).unwrap();
        let data: Vec<_> = (0..6).map(|i| (i + 2) as f32).collect();
        let b = OwnedTensor::new(data, vec![3, 2]).unwrap();
        let mut c = OwnedTensor::zeros(vec![2, 2]);
        matmul(&a, &b, &mut c).unwrap();
        assert_eq!(c.data(), &[16., 19., 52., 64.]);

        let data: Vec<_> = (0..12).map(|i| i as f32).collect();
        let a = OwnedTensor::new(data, vec![2, 2, 3]).unwrap();
        let data: Vec<_> = (0..12).map(|i| (i + 2) as f32).collect();
        let b = OwnedTensor::new(data, vec![2, 3, 2]).unwrap();
        let mut c = OwnedTensor::zeros(vec![2, 2, 2]);
        matmul(&a, &b, &mut c).unwrap();
        assert_eq!(c.data(), &[16., 19., 52., 64., 214., 235., 304., 334.]);
    }

    #[test]
    fn simple_matmul_t() {
        let a = OwnedTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        // A.T
        let b = ViewTensor::new(&[1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let mut c = OwnedTensor::zeros(vec![2, 2]);

        matmul_t(&a, &b, &mut c).unwrap();
        assert_eq!(c.data(), &[7.0, 10.0, 15.0, 22.0]);

        let a = OwnedTensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let b = ViewTensor::new(&[3.0, 4.0], vec![2, 1]).unwrap();
        let mut c = OwnedTensor::zeros(vec![2, 2]);
        matmul_t(&a, &b, &mut c).unwrap();
        assert_eq!(c.data(), &[3.0, 4.0, 6.0, 8.0]);

        let data: Vec<_> = (0..6).map(|i| i as f32).collect();
        let a = OwnedTensor::new(data, vec![2, 3]).unwrap();
        let data: Vec<_> = (0..6).map(|i| (i + 2) as f32).collect();
        let b = OwnedTensor::new(data, vec![2, 3]).unwrap();
        let mut c = OwnedTensor::zeros(vec![2, 2]);
        matmul_t(&a, &b, &mut c).unwrap();
        assert_eq!(c.data(), &[11., 20., 38., 74.]);

        let data: Vec<_> = (0..12).map(|i| i as f32).collect();
        let a = OwnedTensor::new(data, vec![2, 2, 3]).unwrap();
        let data: Vec<_> = (0..12).map(|i| (i + 2) as f32).collect();
        let b = OwnedTensor::new(data, vec![2, 2, 3]).unwrap();
        let mut c = OwnedTensor::zeros(vec![2, 2, 2]);
        matmul_t(&a, &b, &mut c).unwrap();
        assert_eq!(c.data(), &[11., 20., 38., 74., 191., 254., 272., 362.]);
    }

    #[test]
    fn simple_softmax() {
        let mut a = OwnedTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let mut max = vec![0.0; 2];
        softmax(&mut a, &mut max).unwrap();
        assert_eq!(
            simplify(a.data()),
            // Values obtained through python
            [0.2689, 0.7311, 0.2689, 0.7311]
        );
    }

    #[test]
    fn simple_causal_softmax() {
        let mut a = OwnedTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        // Large enough for the second test
        let mut max = vec![0.0; 3 * 2];
        causal_softmax(&mut a, &mut max, 0).unwrap();
        assert_eq!(
            simplify(a.data()),
            // Values obtained through python
            [1.0000, 0.0000, 0.2689, 0.7311]
        );

        let mut a = OwnedTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        causal_softmax(&mut a, &mut max, 1).unwrap();
        assert_eq!(
            simplify(a.data()),
            // Values obtained through python
            [0.2689, 0.7311, 0.2689, 0.7311]
        );

        let data: Vec<_> = (0..12).map(|i| (i + 1) as f32).collect();
        let mut a = OwnedTensor::new(data, vec![3, 2, 2]).unwrap();
        causal_softmax(&mut a, &mut max, 0).unwrap();
        assert_eq!(
            simplify(a.data()),
            // Values obtained through python
            [
                1.0000, 0.0000, 0.2689, 0.7311, 1.0000, 0.0000, 0.2689, 0.7311, 1.0000, 0.0000,
                0.2689, 0.7311
            ]
        );

        let data: Vec<_> = (0..12).map(|i| (i + 1) as f32).collect();
        let mut a = OwnedTensor::new(data, vec![2, 2, 3]).unwrap();
        causal_softmax(&mut a, &mut max, 1).unwrap();
        assert_eq!(
            simplify(a.data()),
            // Values obtained through python
            [
                0.2689, 0.7311, 0.0, 0.09, 0.2447, 0.6652, 0.2689, 0.7311, 0.0, 0.09, 0.2447,
                0.6652
            ]
        );
    }

    #[test]
    fn simple_select() {
        let a = ViewTensor::new(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let mut tensor = OwnedTensor::new(vec![0.0; 6], vec![3, 2]).unwrap();
        select(&[1, 0, 0], &a, &mut tensor).unwrap();
        assert_eq!(
            simplify(tensor.data()),
            // Values obtained through python
            [3.0, 4.0, 1.0, 2.0, 1.0, 2.0]
        );
    }

    #[test]
    fn simple_normalize() {
        let mut a = OwnedTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let epsilon = 1e-5;
        normalize(&mut a, epsilon).unwrap();
        assert_eq!(
            simplify(a.data()),
            // Values obtained through python
            [-1.0, 1.0, -1.0, 1.0]
        );
    }
}
