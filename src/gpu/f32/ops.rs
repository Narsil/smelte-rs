use crate::gpu::f32::Tensor;
use crate::SmeltError;
use cudarc::cublas::result::CublasError;
use cudarc::cublas::safe::{CudaBlas, GemmConfig, StridedBatchedConfig};
use cudarc::cublas::sys::cublasOperation_t::{CUBLAS_OP_N as NoTr, CUBLAS_OP_T as Tr};
use cudarc::cublas::Gemm;
use cudarc::driver::DriverError;
use cudarc::driver::LaunchConfig;
use cudarc::driver::DeviceSlice;
use cudarc::driver::LaunchAsync;

/// All potential errors linked specifically to cuda.
#[derive(Debug, Clone)]
pub enum CudaError {
    /// Tried an operation with tensors on different devices.
    TensorOnDifferentDevice {
        /// TODO
        got: usize,
        /// TODO
        expected: usize,
    },
    /// Error with cublas library
    CublasError(CublasError),
    /// Error with cuda driver.
    DriverError(DriverError),
}

impl From<CublasError> for SmeltError {
    fn from(cublas: CublasError) -> Self {
        Self::Cuda(CudaError::CublasError(cublas))
    }
}

impl From<DriverError> for SmeltError {
    fn from(cublas: DriverError) -> Self {
        Self::Cuda(CudaError::DriverError(cublas))
    }
}
// use crate::gpu::f32::tensor::Tensor;
// use crate::SmeltError;
//
// /// Operation for selecting entire rows within tensor `weights`. Each `id` is the index
// /// of the row.
// pub fn select(ids: &[usize], weights: &Tensor, out: &mut Tensor) -> Result<(), SmeltError> {
//     let sequence_length = ids.len();
//     let vocab_size = weights.shape()[0];
//     let hidden_dim = weights.shape()[1];
//     if out.shape() != [sequence_length, hidden_dim] {
//         return Err(SmeltError::DimensionMismatch {
//             expected: vec![sequence_length, hidden_dim],
//             got: out.shape().to_vec(),
//         });
//     }
//     for (i, id) in ids.iter().enumerate() {
//         let id = *id;
//         if id >= vocab_size {
//             return Err(SmeltError::OutOfVocabulary { vocab_size, id });
//         }
//         let weight_offset = id * hidden_dim;
//         let data_offset = i * hidden_dim;
//         out.data_mut()[data_offset..data_offset + hidden_dim]
//             .copy_from_slice(&weights.data()[weight_offset..weight_offset + hidden_dim]);
//     }
//     Ok(())
// }
//
/// Regular matrix multiplication
pub fn matmul(a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<(), SmeltError> {
    g_matmul::<false>(a, b, out)
}

/// Matrix multiplication matmul(A, B.transposed())
pub fn matmul_t(a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<(), SmeltError> {
    g_matmul::<true>(a, b, out)
}

#[inline]
fn g_matmul<'a, const TRANSPOSE: bool>(
    a: &Tensor,
    b: &Tensor,
    c: &mut Tensor,
) -> Result<(), SmeltError> {
    let dim = a.shape().len();

    if a.device_id() != b.device_id() {
        return Err(SmeltError::Cuda(CudaError::TensorOnDifferentDevice {
            got: b.device_id(),
            expected: a.device_id(),
        }));
    }
    if a.device_id() != c.device_id() {
        return Err(SmeltError::Cuda(CudaError::TensorOnDifferentDevice {
            got: c.device_id(),
            expected: a.device_id(),
        }));
    }

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

    // TODO Maybe Zero out c
    // c.data_mut().iter_mut().for_each(|v| *v = 0.0);
    c.device().memset_zeros(c.data_mut())?;

    let batching: usize = a.shape()[..dim - 2].iter().product();
    let a_skip: usize = m * k;
    let b_skip: usize = n * k;
    let c_skip: usize = m * n;

    let blas = CudaBlas::new(a.device())?;

    let (m, n, k) = (m as libc::c_int, n as libc::c_int, k as libc::c_int);

    // Swap everything around
    // Cublas uses col major format, so it will read
    // A as A.T
    // B as B.t
    // So we calculate in C.T <- matmul(B.t, A.t)
    // But since C.t in read by us in row major, it is already C.

    let (m, n, k) = (n, m, k);
    let (a_skip, b_skip) = (b_skip, a_skip);
    let (a, b) = (b, a);

    let (ldb, ldc) = (k, m);
    let (lda, transa) = if TRANSPOSE { (k, Tr) } else { (m, NoTr) };

    let transb = NoTr;

    let cfg = GemmConfig {
        transa,
        transb,
        m,
        n,
        k,
        alpha: 1.0,
        lda,
        ldb,
        beta: 1.0,
        ldc,
    };

    let strided_config = StridedBatchedConfig {
        gemm: cfg,
        batch_size: batching as i32,
        stride_a: a_skip as i64,
        stride_b: b_skip as i64,
        stride_c: c_skip as i64,
    };
    unsafe {
        blas.gemm_strided_batched(strided_config, &*a.data(), &*b.data(), &mut *c.data_mut())?;
    }

    Ok(())
}

const ADD_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/add.ptx"));
/// tensor elementwise addition. b += a.
pub fn add(a: &Tensor, b: &mut Tensor) -> Result<(), SmeltError> {
    if a.shape() != b.shape() {
        return Err(SmeltError::DimensionMismatch {
            expected: b.shape().to_vec(),
            got: a.shape().to_vec(),
        });
    }
    if a.device_id() != b.device_id() {
        return Err(SmeltError::Cuda(CudaError::TensorOnDifferentDevice {
            got: b.device_id(),
            expected: a.device_id(),
        }));
    }

    let dev = a.device();

    let module_name = "add_fwd_f32";
     if !dev.has_func(module_name, module_name) {
        dev
            .load_ptx(ADD_PTX.into(), module_name, &[module_name])?;
    }

    let numel = a.data().len();

    let fwd_fn = dev.get_func(module_name, module_name).unwrap();
    let cfg = LaunchConfig::for_num_elems(numel as u32);
    let params = (numel, a.data(), b.data_mut());
    unsafe { fwd_fn.launch(cfg, params) }?;

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
    let skip: usize = a.shape().iter().product();
    if a.device_id() != b.device_id() {
        return Err(SmeltError::Cuda(CudaError::TensorOnDifferentDevice {
            got: b.device_id(),
            expected: a.device_id(),
        }));
    }

    let dev = a.device();

    let module_name = "badd_fwd_f32";
     if !dev.has_func(module_name, module_name) {
        dev
            .load_ptx(ADD_PTX.into(), module_name, &[module_name])?;
    }

    let numel = b.data().len();

    let fwd_fn = dev.get_func(module_name, module_name).unwrap();
    let cfg = LaunchConfig::for_num_elems(numel as u32);
    let params = (numel, a.data(), b.data_mut(), skip);
    unsafe { fwd_fn.launch(cfg, params) }?;

    Ok(())
}

/// tensor elementwise multiplication. b *= a.
pub fn mul(a: &Tensor, b: &mut Tensor) -> Result<(), SmeltError> {
    if a.shape() != b.shape() {
        return Err(SmeltError::DimensionMismatch {
            expected: b.shape().to_vec(),
            got: a.shape().to_vec(),
        });
    }
    if a.device_id() != b.device_id() {
        return Err(SmeltError::Cuda(CudaError::TensorOnDifferentDevice {
            got: b.device_id(),
            expected: a.device_id(),
        }));
    }

    let dev = a.device();

    let module_name = "mul_fwd_f32";
     if !dev.has_func(module_name, module_name) {
        dev
            .load_ptx(ADD_PTX.into(), module_name, &[module_name])?;
    }

    let numel = a.data().len();

    let fwd_fn = dev.get_func(module_name, module_name).unwrap();
    let cfg = LaunchConfig::for_num_elems(numel as u32);
    let params = (numel, a.data(), b.data_mut());
    unsafe { fwd_fn.launch(cfg, params) }?;

    Ok(())
}


 
/// broadcasted tensor elementwise multiplication. b *= a.
pub fn broadcast_mul(a: &Tensor, b: &mut Tensor) -> Result<(), SmeltError> {
    if &b.shape()[1..] != a.shape() {
        return Err(SmeltError::DimensionMismatch {
            expected: b.shape().to_vec(),
            got: a.shape().to_vec(),
        });
    }
    let skip: usize = a.shape().iter().product();
    if a.device_id() != b.device_id() {
        return Err(SmeltError::Cuda(CudaError::TensorOnDifferentDevice {
            got: b.device_id(),
            expected: a.device_id(),
        }));
    }

    let dev = a.device();

    let module_name = "bmul_fwd_f32";
     if !dev.has_func(module_name, module_name) {
        dev
            .load_ptx(ADD_PTX.into(), module_name, &[module_name])?;
    }

    let numel = b.data().len();

    let fwd_fn = dev.get_func(module_name, module_name).unwrap();
    let cfg = LaunchConfig::for_num_elems(numel as u32);
    let params = (numel, a.data(), b.data_mut(), skip);
    unsafe { fwd_fn.launch(cfg, params) }?;

    Ok(())
}


// /// Basic operation for the layernorm.
// /// x = (x - x.mean()) / (x.var() + epsilon)
// /// `mean` and `var` do not have to be initialized, they are simply passed to
// /// avoid allocation.
// pub fn normalize(x: &mut Tensor, epsilon: f32) -> Result<(), SmeltError> {
//     let dim = x.shape().len();
//     let size = x.shape()[dim - 1];
//     x.data_mut().chunks_mut(size).for_each(|chunk| {
//         let sum: f32 = chunk.iter().sum();
//         let mean = sum / size as f32;
//         chunk.iter_mut().for_each(|v| *v -= mean);
//         let var: f32 = chunk.iter().map(|v| v * v).sum();
//         let var = var / size as f32;
//         let stddev: f32 = (var + epsilon).sqrt();
//         chunk.iter_mut().for_each(|v| *v /= stddev);
//     });
//     Ok(())
// }
//
// #[inline]
// fn g_softmax<const CAUSAL: bool>(
//     x: &mut Tensor,
//     past_sequence_length: usize,
// ) -> Result<(), SmeltError> {
//     let dim = x.shape().len();
//
//     let m = x.shape()[dim - 2];
//     let n = x.shape()[dim - 1];
//
//     x.data_mut()
//         .chunks_mut(n)
//         .enumerate()
//         .for_each(|(i, chunk)| {
//             let i = i % m;
//             let mut current_max = f32::NEG_INFINITY;
//             for (j, &v) in chunk.iter().enumerate() {
//                 if (!CAUSAL || i + past_sequence_length >= j) && v > current_max {
//                     current_max = v;
//                 }
//             }
//             for v in chunk.iter_mut() {
//                 *v -= current_max;
//                 *v = (*v).exp();
//             }
//             let mut sum = 0.0;
//             for (j, &v) in chunk.iter().enumerate() {
//                 if !CAUSAL || i + past_sequence_length >= j {
//                     sum += v;
//                 }
//             }
//             for (j, v) in chunk.iter_mut().enumerate() {
//                 if !CAUSAL || i + past_sequence_length >= j {
//                     *v /= sum;
//                 } else {
//                     *v = 0.0;
//                 }
//             }
//         });
//     Ok(())
// }
//
// /// Softmax on the last dimension for tensor `x`
// pub fn softmax(x: &mut Tensor) -> Result<(), SmeltError> {
//     g_softmax::<false>(x, 0)
// }
//
// /// Causal softmax on the last dimension for tensor `x`. The causality is determined by the
// /// shape of `x` and `past_sequence_length` which defines how big is the missing part of the
// /// square.
// pub fn causal_softmax(x: &mut Tensor, past_sequence_length: usize) -> Result<(), SmeltError> {
//     g_softmax::<true>(x, past_sequence_length)
// }
//
// /// Argmax of the last dimension of tensor `x `.
// pub fn special_argmax(x: &Tensor) -> Result<usize, SmeltError> {
//     if x.shape().len() != 2 {
//         return Err(SmeltError::InvalidRank { expected_rank: 2 });
//     }
//     let n = x.shape()[0];
//     let m = x.shape()[1];
//
//     let mut max = f32::NEG_INFINITY;
//     let mut max_id = usize::MAX;
//     for (i, &v) in x.data().iter().skip((n - 1) * m).enumerate() {
//         if v > max {
//             max = v;
//             max_id = i;
//         }
//     }
//     Ok(max_id)
// }
//
// /// utility function to use a faster but less precise tanh
// pub fn faster_tanh(x: f32) -> f32 {
//     let x2 = x * x;
//     let x3 = x2 * x;
//     let x5 = x3 * x2;
//
//     let a = x + (0.16489087 * x3) + (0.00985468 * x5);
//
//     a / (1.0 + (a * a)).sqrt()
// }
//
// /// utility function to use a faster but less precise tanh
// #[inline]
// pub fn inline_tanh(x: f32) -> f32 {
//     1.0 - (2.0 / (1.0 + (2.0 * x).exp()))
// }
//
// /// `gelu` operation
// /// <https://en.wikipedia.org/wiki/Activation_function#Comparison_of_activation_functions>
// /// but using [faster_tanh]
// #[inline]
// pub fn faster_gelu(v: f32) -> f32 {
//     0.5 * (v)
//         * (1.0 + faster_tanh((2.0f32 / std::f32::consts::PI).sqrt() * v * (1.0 + 0.044715 * v * v)))
// }
//
// /// `gelu` operation
// /// <https://en.wikipedia.org/wiki/Activation_function#Comparison_of_activation_functions>
// #[inline]
// pub fn gelu(v: f32) -> f32 {
//     0.5 * (v)
//         * (1.0 + inline_tanh((2.0f32 / std::f32::consts::PI).sqrt() * v * (1.0 + 0.044715 * v * v)))
// }
//
// /// Applies `func` to every item of the tensor
// pub fn apply<F: Fn(f32) -> f32 + Sync>(x: &mut Tensor, func: F) {
//     x.data_mut().iter_mut().for_each(|v| *v = func(*v));
// }
//
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::simplify;

    #[test]
    fn simple_matmul() {
        let device_id = 0;
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let a = Tensor::from_cpu(data.clone(), vec![2, 2], device_id).unwrap();
        let b = Tensor::from_cpu(data, vec![2, 2], device_id).unwrap();
        let mut c = Tensor::zeros(vec![2, 2], device_id).unwrap();

        matmul(&a, &b, &mut c).unwrap();
        assert_eq!(c.cpu_data().unwrap(), &[7.0, 10.0, 15.0, 22.0]);
        matmul(&a, &b, &mut c).unwrap();
        assert_eq!(c.cpu_data().unwrap(), &[7.0, 10.0, 15.0, 22.0]);

        let data = vec![1.0, 2.0, 3.0];
        let a = Tensor::from_cpu(data, vec![3, 1], device_id).unwrap();
        let b = Tensor::from_cpu(vec![3.0, 4.0], vec![1, 2], device_id).unwrap();
        let mut c = Tensor::zeros(vec![3, 2], device_id).unwrap();
        matmul(&a, &b, &mut c).unwrap();
        assert_eq!(c.cpu_data().unwrap(), &[3.0, 4.0, 6.0, 8.0, 9.0, 12.0]);

        let data: Vec<_> = (0..6).map(|i| i as f32).collect();
        let a = Tensor::from_cpu(data, vec![2, 3], device_id).unwrap();
        let data: Vec<_> = (0..12).map(|i| (i + 2) as f32).collect();
        let b = Tensor::from_cpu(data, vec![3, 4], device_id).unwrap();
        let mut c = Tensor::zeros(vec![2, 4], device_id).unwrap();
        matmul(&a, &b, &mut c).unwrap();
        assert_eq!(
            c.cpu_data().unwrap(),
            &[26., 29., 32., 35., 80., 92., 104., 116.]
        );

        let data: Vec<_> = (0..12).map(|i| i as f32).collect();
        let a = Tensor::from_cpu(data, vec![2, 2, 3], device_id).unwrap();
        let data: Vec<_> = (0..24).map(|i| (i + 2) as f32).collect();
        let b = Tensor::from_cpu(data, vec![2, 3, 4], device_id).unwrap();
        let mut c: Tensor = Tensor::zeros(vec![2, 2, 4], device_id).unwrap();
        matmul(&a, &b, &mut c).unwrap();
        assert_eq!(
            c.cpu_data().unwrap(),
            &[
                26., 29., 32., 35., 80., 92., 104., 116., 386., 407., 428., 449., 548., 578., 608.,
                638.
            ]
        );
    }

    #[test]
    fn simple_matmul_t() {
        let device_id = 0;
        let a = Tensor::from_cpu(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], device_id).unwrap();
        // A.T
        let b = Tensor::from_cpu(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2], device_id).unwrap();
        let mut c = Tensor::zeros(vec![2, 2], device_id).unwrap();

        matmul_t(&a, &b, &mut c).unwrap();
        assert_eq!(c.cpu_data().unwrap(), &[7.0, 10.0, 15.0, 22.0]);

        let a = Tensor::from_cpu(vec![1.0, 2.0], vec![2, 1], device_id).unwrap();
        let b = Tensor::from_cpu(vec![3.0, 4.0], vec![2, 1], device_id).unwrap();
        let mut c = Tensor::zeros(vec![2, 2], device_id).unwrap();
        matmul_t(&a, &b, &mut c).unwrap();
        assert_eq!(c.cpu_data().unwrap(), &[3.0, 4.0, 6.0, 8.0]);

        let data: Vec<_> = (0..6).map(|i| i as f32).collect();
        let a = Tensor::from_cpu(data, vec![2, 3], device_id).unwrap();
        let data: Vec<_> = (0..12).map(|i| (i + 2) as f32).collect();
        let b = Tensor::from_cpu(data, vec![4, 3], device_id).unwrap();
        let mut c = Tensor::zeros(vec![2, 4], device_id).unwrap();
        matmul_t(&a, &b, &mut c).unwrap();
        assert_eq!(
            c.cpu_data().unwrap(),
            &[11., 20., 29., 38., 38., 74., 110., 146.]
        );

        let data: Vec<_> = (0..12).map(|i| i as f32).collect();
        let a = Tensor::from_cpu(data, vec![2, 2, 3], device_id).unwrap();
        let data: Vec<_> = (0..24).map(|i| (i + 2) as f32).collect();
        let b = Tensor::from_cpu(data, vec![2, 4, 3], device_id).unwrap();
        let mut c = Tensor::zeros(vec![2, 2, 4], device_id).unwrap();
        matmul_t(&a, &b, &mut c).unwrap();
        assert_eq!(
            c.cpu_data().unwrap(),
            &[
                11., 20., 29., 38., 38., 74., 110., 146., 317., 380., 443., 506., 452., 542., 632.,
                722.
            ]
        );
    }
    
    #[test]
    fn simple_add() {
        let a = Tensor::from_cpu(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], 0).unwrap();
        let mut b = Tensor::from_cpu(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2], 0).unwrap();
        add(&a, &mut b).unwrap();
        assert_eq!(
            b.cpu_data().unwrap(),
            // Values obtained through python
            [2.0, 3.0, 4.0, 5.0]
        );
    }

    #[test]
    fn simple_broadcast_add() {
        let a = Tensor::from_cpu(vec![1.0, 2.0], vec![2], 0).unwrap();
        let mut b = Tensor::from_cpu(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0], vec![3, 2], 0).unwrap();
        broadcast_add(&a, &mut b).unwrap();
        assert_eq!(
            b.cpu_data().unwrap(),
            // Values obtained through python
            [2.0, 3.0, 2.0, 3.0, 2.0, 3.0]
        );
    }

    #[test]
    fn simple_mul() {
        let a = Tensor::from_cpu(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], 0).unwrap();
        let mut b = Tensor::from_cpu(vec![2.0, 2.0, 3.0, 3.0], vec![2, 2], 0).unwrap();
        mul(&a, &mut b).unwrap();
        assert_eq!(
            b.cpu_data().unwrap(),
            // Values obtained through python
            [2.0, 4.0, 9.0, 12.0]
        );
    }

    //
    //     #[test]
    //     fn simple_softmax() {
    //         let mut a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    //         softmax(&mut a).unwrap();
    //         assert_eq!(
    //             simplify(a.data()),
    //             // Values obtained through python
    //             [0.2689, 0.7311, 0.2689, 0.7311]
    //         );
    //     }
    //
    //     #[test]
    //     fn simple_causal_softmax() {
    //         let mut a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    //         // Large enough for the second test
    //         causal_softmax(&mut a, 0).unwrap();
    //         assert_eq!(
    //             simplify(a.data()),
    //             // Values obtained through python
    //             [1.0000, 0.0000, 0.2689, 0.7311]
    //         );
    //
    //         let mut a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    //         causal_softmax(&mut a, 1).unwrap();
    //         assert_eq!(
    //             simplify(a.data()),
    //             // Values obtained through python
    //             [0.2689, 0.7311, 0.2689, 0.7311]
    //         );
    //
    //         let data: Vec<_> = (0..12).map(|i| (i + 1) as f32).collect();
    //         let mut a = Tensor::new(data, vec![3, 2, 2]).unwrap();
    //         causal_softmax(&mut a, 0).unwrap();
    //         assert_eq!(
    //             simplify(a.data()),
    //             // Values obtained through python
    //             [
    //                 1.0000, 0.0000, 0.2689, 0.7311, 1.0000, 0.0000, 0.2689, 0.7311, 1.0000, 0.0000,
    //                 0.2689, 0.7311
    //             ]
    //         );
    //
    //         let data: Vec<_> = (0..12).map(|i| (i + 1) as f32).collect();
    //         let mut a = Tensor::new(data, vec![2, 2, 3]).unwrap();
    //         causal_softmax(&mut a, 1).unwrap();
    //         assert_eq!(
    //             simplify(a.data()),
    //             // Values obtained through python
    //             [
    //                 0.2689, 0.7311, 0.0, 0.09, 0.2447, 0.6652, 0.2689, 0.7311, 0.0, 0.09, 0.2447,
    //                 0.6652
    //             ]
    //         );
    //     }
    //
    //     #[test]
    //     fn simple_select() {
    //         let a = Tensor::borrowed(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    //         let mut tensor = Tensor::zeros(vec![3, 2]);
    //         select(&[1, 0, 0], &a, &mut tensor).unwrap();
    //         assert_eq!(
    //             simplify(tensor.data()),
    //             // Values obtained through python
    //             [3.0, 4.0, 1.0, 2.0, 1.0, 2.0]
    //         );
    //     }
    //
    //     #[test]
    //     fn simple_normalize() {
    //         let mut a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    //         let epsilon = 1e-5;
    //         normalize(&mut a, epsilon).unwrap();
    //         assert_eq!(
    //             simplify(a.data()),
    //             // Values obtained through python
    //             [-1.0, 1.0, -1.0, 1.0]
    //         );
    //     }
}
