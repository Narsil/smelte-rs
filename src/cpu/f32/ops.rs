use crate::cpu::f32::tensor::Tensor;
use crate::SmeltError;

#[cfg(feature = "matrixmultiply")]
use matrixmultiply::sgemm;

#[cfg(feature = "cblas")]
use cblas_sys::{
    cblas_sgemm as sgemm, CblasColMajor as ColMajor, CblasNoTrans as NoTr,
    CblasRowMajor as RowMajor, CblasTrans as Tr,
};

/// Operation for selecting entire rows within tensor `weights`. Each `id` is the index
/// of the row.
pub fn select(ids: &[usize], weights: &Tensor, out: &mut Tensor) -> Result<(), SmeltError> {
    let sequence_length = ids.len();
    let vocab_size = weights.shape()[0];
    let hidden_dim = weights.shape()[1];
    if out.shape() != [sequence_length, hidden_dim] {
        return Err(SmeltError::DimensionMismatch {
            expected: vec![sequence_length, hidden_dim],
            got: out.shape().to_vec(),
        });
    }
    for (i, id) in ids.iter().enumerate() {
        let id = *id;
        if id >= vocab_size {
            return Err(SmeltError::OutOfVocabulary { vocab_size, id });
        }
        let weight_offset = id * hidden_dim;
        let data_offset = i * hidden_dim;
        out.data_mut()[data_offset..data_offset + hidden_dim]
            .copy_from_slice(&weights.data()[weight_offset..weight_offset + hidden_dim]);
    }
    Ok(())
}

/// Regular matrix multiplication
pub fn matmul<'a>(a: &Tensor<'a>, b: &Tensor<'a>, out: &mut Tensor<'a>) -> Result<(), SmeltError> {
    g_matmul::<false>(a, b, out)
}

/// Matrix multiplication matmul(A, B.transposed())
pub fn matmul_t<'a>(
    a: &Tensor<'a>,
    b: &Tensor<'a>,
    out: &mut Tensor<'a>,
) -> Result<(), SmeltError> {
    g_matmul::<true>(a, b, out)
}

#[inline]
fn g_matmul<'a, const TRANSPOSE: bool>(
    a: &Tensor<'a>,
    b: &Tensor<'a>,
    c: &mut Tensor<'a>,
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

    // Zero out c
    c.data_mut().iter_mut().for_each(|v| *v = 0.0);

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

    (0..batching).for_each(|step| {
        let ap = &a.data()[step * a_skip..];
        let bp = &b.data()[step * b_skip..];
        let cp = &mut c.data_mut()[step * c_skip..];

        #[cfg(feature = "matrixmultiply")]
        unsafe {
            sgemm(
                m,
                k,
                n,
                1.0,
                ap.as_ptr(),
                ar,
                ac,
                bp.as_ptr(),
                br,
                bc,
                1.0,
                cp.as_mut_ptr(),
                cr,
                cc,
            );
        }

        #[cfg(feature = "cblas")]
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
            sgemm(
                layout,
                a_tr,
                b_tr,
                m,
                n,
                k,
                1.0,
                ap.as_ptr(),
                lda,
                // a_skip as i32,
                bp.as_ptr(),
                ldb,
                // b_skip as i32,
                1.0,
                cp.as_mut_ptr(),
                ldc,
                // c_skip as i32,
                // batching as i32,
            )
        }
    });
    Ok(())
}

/// tensor elementwise addition. b += a.
pub fn add(a: &Tensor, b: &mut Tensor) -> Result<(), SmeltError> {
    if a.shape() != b.shape() {
        return Err(SmeltError::DimensionMismatch {
            expected: b.shape().to_vec(),
            got: a.shape().to_vec(),
        });
    }
    a.data()
        .iter()
        .zip(b.data_mut().iter_mut())
        .for_each(|(left, right)| *right += left);
    Ok(())
}

/// broacasted tensor elementwise addition. b += a.
pub fn broadcast_add(a: &Tensor, b: &mut Tensor) -> Result<(), SmeltError> {
    if &b.shape()[1..] != a.shape() {
        return Err(SmeltError::DimensionMismatch {
            expected: b.shape().to_vec(),
            got: a.shape().to_vec(),
        });
    }
    let n = b.shape()[0];
    let skip: usize = a.shape().iter().product();
    (0..n).for_each(|i| {
        a.data()
            .iter()
            .zip(b.data_mut().iter_mut().skip(i * skip))
            .for_each(|(left, right)| *right += left);
    });
    Ok(())
}

/// tensor elementwise multiplication. b += a.
pub fn mul(a: &Tensor, b: &mut Tensor) -> Result<(), SmeltError> {
    if a.shape() != b.shape() {
        return Err(SmeltError::DimensionMismatch {
            expected: b.shape().to_vec(),
            got: a.shape().to_vec(),
        });
    }
    a.data()
        .iter()
        .zip(b.data_mut().iter_mut())
        .for_each(|(left, right)| *right *= left);
    Ok(())
}

/// broacasted tensor elementwise multiplication. b += a.
pub fn broadcast_mul(a: &Tensor, b: &mut Tensor) -> Result<(), SmeltError> {
    if &b.shape()[1..] != a.shape() {
        return Err(SmeltError::DimensionMismatch {
            expected: b.shape().to_vec(),
            got: a.shape().to_vec(),
        });
    }
    let n = b.shape()[0];
    let skip: usize = a.shape().iter().product();
    (0..n).for_each(|i| {
        a.data()
            .iter()
            .zip(b.data_mut().iter_mut().skip(i * skip))
            .for_each(|(left, right)| *right *= left);
    });
    Ok(())
}

/// Basic operation for the layernorm.
/// x = (x - x.mean()) / (x.var() + epsilon)
/// `mean` and `var` do not have to be initialized, they are simply passed to
/// avoid allocation.
pub fn normalize(x: &mut Tensor, epsilon: f32) -> Result<(), SmeltError> {
    let dim = x.shape().len();
    let size = x.shape()[dim - 1];
    x.data_mut().chunks_mut(size).for_each(|chunk| {
        let sum: f32 = chunk.iter().sum();
        let mean = sum / size as f32;
        chunk.iter_mut().for_each(|v| *v -= mean);
        let var: f32 = chunk.iter().map(|v| v * v).sum();
        let var = var / size as f32;
        let stddev: f32 = (var + epsilon).sqrt();
        chunk.iter_mut().for_each(|v| *v /= stddev);
    });
    Ok(())
}

#[inline]
fn g_softmax<const CAUSAL: bool>(
    x: &mut Tensor,
    past_sequence_length: usize,
) -> Result<(), SmeltError> {
    let dim = x.shape().len();

    let m = x.shape()[dim - 2];
    let n = x.shape()[dim - 1];

    x.data_mut()
        .chunks_mut(n)
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
pub fn softmax(x: &mut Tensor) -> Result<(), SmeltError> {
    g_softmax::<false>(x, 0)
}

/// Causal softmax on the last dimension for tensor `x`. The causality is determined by the
/// shape of `x` and `past_sequence_length` which defines how big is the missing part of the
/// square.
pub fn causal_softmax(x: &mut Tensor, past_sequence_length: usize) -> Result<(), SmeltError> {
    g_softmax::<true>(x, past_sequence_length)
}

/// Argmax of the last dimension of tensor `x `.
pub fn special_argmax(x: &Tensor) -> Result<usize, SmeltError> {
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

/// utility function to use a faster but less precise tanh
#[inline]
pub fn inline_tanh(x: f32) -> f32 {
    1.0 - (2.0 / (1.0 + (2.0 * x).exp()))
}

/// `gelu` operation
/// <https://en.wikipedia.org/wiki/Activation_function#Comparison_of_activation_functions>
/// but using [faster_tanh]
#[inline]
pub fn faster_gelu(v: f32) -> f32 {
    0.5 * (v)
        * (1.0 + faster_tanh((2.0f32 / std::f32::consts::PI).sqrt() * v * (1.0 + 0.044715 * v * v)))
}

/// `gelu` operation
/// <https://en.wikipedia.org/wiki/Activation_function#Comparison_of_activation_functions>
#[inline]
pub fn gelu(v: f32) -> f32 {
    0.5 * (v)
        * (1.0 + inline_tanh((2.0f32 / std::f32::consts::PI).sqrt() * v * (1.0 + 0.044715 * v * v)))
}

/// Applies `func` to every item of the tensor
pub fn apply<F: Fn(f32) -> f32 + Sync>(x: &mut Tensor, func: F) {
    x.data_mut().iter_mut().for_each(|v| *v = func(*v));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::simplify;

    #[test]
    fn simple_broadcast_add() {
        let a = Tensor::new(vec![1.0, 2.0], vec![2]).unwrap();
        let mut b = Tensor::new(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0], vec![3, 2]).unwrap();
        broadcast_add(&a, &mut b).unwrap();
        assert_eq!(b.data(), [2.0, 3.0, 2.0, 3.0, 2.0, 3.0]);
    }

    #[test]
    fn simple_matmul() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let a = Tensor::new(data, vec![2, 2]).unwrap();
        let data = [1.0, 2.0, 3.0, 4.0];
        let b = Tensor::borrowed(&data, vec![2, 2]).unwrap();
        let data = vec![0.0; 4];
        let mut c = Tensor::new(data, vec![2, 2]).unwrap();

        matmul(&a, &b, &mut c).unwrap();
        assert_eq!(c.data(), &[7.0, 10.0, 15.0, 22.0]);
        matmul(&a, &b, &mut c).unwrap();
        assert_eq!(c.data(), &[7.0, 10.0, 15.0, 22.0]);

        let data = vec![1.0, 2.0];
        let a = Tensor::new(data, vec![2, 1]).unwrap();
        let data = [3.0, 4.0];
        let b = Tensor::borrowed(&data, vec![1, 2]).unwrap();
        let data = vec![0.0; 4];
        let mut c = Tensor::new(data, vec![2, 2]).unwrap();
        matmul(&a, &b, &mut c).unwrap();
        assert_eq!(c.data(), &[3.0, 4.0, 6.0, 8.0]);

        let data: Vec<_> = (0..6).map(|i| i as f32).collect();
        let a = Tensor::new(data, vec![2, 3]).unwrap();
        let data: Vec<_> = (0..6).map(|i| (i + 2) as f32).collect();
        let b = Tensor::new(data, vec![3, 2]).unwrap();
        let mut c = Tensor::zeros(vec![2, 2]);
        matmul(&a, &b, &mut c).unwrap();
        assert_eq!(c.data(), &[16., 19., 52., 64.]);

        let data: Vec<_> = (0..12).map(|i| i as f32).collect();
        let a = Tensor::new(data, vec![2, 2, 3]).unwrap();
        let data: Vec<_> = (0..12).map(|i| (i + 2) as f32).collect();
        let b = Tensor::new(data, vec![2, 3, 2]).unwrap();
        let mut c: Tensor = Tensor::zeros(vec![2, 2, 2]);
        matmul(&a, &b, &mut c).unwrap();
        assert_eq!(c.data(), &[16., 19., 52., 64., 214., 235., 304., 334.]);
    }

    #[test]
    fn simple_matmul_t() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        // A.T
        let b = Tensor::borrowed(&[1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let mut c = Tensor::zeros(vec![2, 2]);

        matmul_t(&a, &b, &mut c).unwrap();
        assert_eq!(c.data(), &[7.0, 10.0, 15.0, 22.0]);

        let a = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let b = Tensor::borrowed(&[3.0, 4.0], vec![2, 1]).unwrap();
        let mut c = Tensor::zeros(vec![2, 2]);
        matmul_t(&a, &b, &mut c).unwrap();
        assert_eq!(c.data(), &[3.0, 4.0, 6.0, 8.0]);

        let data: Vec<_> = (0..6).map(|i| i as f32).collect();
        let a = Tensor::new(data, vec![2, 3]).unwrap();
        let data: Vec<_> = (0..6).map(|i| (i + 2) as f32).collect();
        let b = Tensor::new(data, vec![2, 3]).unwrap();
        let mut c = Tensor::zeros(vec![2, 2]);
        matmul_t(&a, &b, &mut c).unwrap();
        assert_eq!(c.data(), &[11., 20., 38., 74.]);

        let data: Vec<_> = (0..12).map(|i| i as f32).collect();
        let a = Tensor::new(data, vec![2, 2, 3]).unwrap();
        let data: Vec<_> = (0..12).map(|i| (i + 2) as f32).collect();
        let b = Tensor::new(data, vec![2, 2, 3]).unwrap();
        let mut c = Tensor::zeros(vec![2, 2, 2]);
        matmul_t(&a, &b, &mut c).unwrap();
        assert_eq!(c.data(), &[11., 20., 38., 74., 191., 254., 272., 362.]);
    }

    #[test]
    fn simple_softmax() {
        let mut a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        softmax(&mut a).unwrap();
        assert_eq!(
            simplify(a.data()),
            // Values obtained through python
            [0.2689, 0.7311, 0.2689, 0.7311]
        );
    }

    #[test]
    fn simple_causal_softmax() {
        let mut a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        // Large enough for the second test
        causal_softmax(&mut a, 0).unwrap();
        assert_eq!(
            simplify(a.data()),
            // Values obtained through python
            [1.0000, 0.0000, 0.2689, 0.7311]
        );

        let mut a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        causal_softmax(&mut a, 1).unwrap();
        assert_eq!(
            simplify(a.data()),
            // Values obtained through python
            [0.2689, 0.7311, 0.2689, 0.7311]
        );

        let data: Vec<_> = (0..12).map(|i| (i + 1) as f32).collect();
        let mut a = Tensor::new(data, vec![3, 2, 2]).unwrap();
        causal_softmax(&mut a, 0).unwrap();
        assert_eq!(
            simplify(a.data()),
            // Values obtained through python
            [
                1.0000, 0.0000, 0.2689, 0.7311, 1.0000, 0.0000, 0.2689, 0.7311, 1.0000, 0.0000,
                0.2689, 0.7311
            ]
        );

        let data: Vec<_> = (0..12).map(|i| (i + 1) as f32).collect();
        let mut a = Tensor::new(data, vec![2, 2, 3]).unwrap();
        causal_softmax(&mut a, 1).unwrap();
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
        let a = Tensor::borrowed(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let mut tensor = Tensor::zeros(vec![3, 2]);
        select(&[1, 0, 0], &a, &mut tensor).unwrap();
        assert_eq!(
            simplify(tensor.data()),
            // Values obtained through python
            [3.0, 4.0, 1.0, 2.0, 1.0, 2.0]
        );
    }

    #[test]
    fn simple_normalize() {
        let mut a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let epsilon = 1e-5;
        normalize(&mut a, epsilon).unwrap();
        assert_eq!(
            simplify(a.data()),
            // Values obtained through python
            [-1.0, 1.0, -1.0, 1.0]
        );

        // TODO Figure out how torch does layernorm to make sure this works.
        // let mut a = Tensor::new(vec![-0.8570, -1.4722, -1.7398, -0.5307, -0.4816,  0.2071], vec![2, 3]).unwrap();
        // let epsilon = 1e-5;
        // normalize(&mut a, epsilon).unwrap();
        // assert_eq!(
        //     simplify(a.data()),
        //     // Values obtained through python
        //     [ 1.1031, -0.2559, -0.8472, -0.6359, -0.5167,  1.1526]
        // );
    }
}
