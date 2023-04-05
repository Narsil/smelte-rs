use crate::gpu::f32::Tensor;
use crate::SmeltError;
use cudarc::cublas::result::CublasError;
use cudarc::cublas::safe::{GemmConfig, StridedBatchedConfig};
use cudarc::cublas::sys::cublasOperation_t::{CUBLAS_OP_N as NoTr, CUBLAS_OP_T as Tr};
use cudarc::cublas::Gemm;
use cudarc::driver::DeviceSlice;
use cudarc::driver::DriverError;
use cudarc::driver::LaunchAsync;
use cudarc::driver::LaunchConfig;

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
    if weights.device_id() != out.device_id() {
        return Err(SmeltError::Cuda(CudaError::TensorOnDifferentDevice {
            got: out.device_id(),
            expected: weights.device_id(),
        }));
    }

    let dev = weights.cuda();

    for (i, id) in ids.iter().enumerate() {
        let id = *id;
        if id >= vocab_size {
            return Err(SmeltError::OutOfVocabulary { vocab_size, id });
        }
        let weight_offset = id * hidden_dim;
        let data_offset = i * hidden_dim;

        let src = weights
            .data()
            .slice(weight_offset..weight_offset + hidden_dim);
        let mut dst = out
            .data_mut()
            .slice_mut(data_offset..data_offset + hidden_dim);
        dev.dtod_copy(&src, &mut dst)?
    }
    Ok(())
}

/// Copy tensor into another tensor
pub fn copy(weights: &Tensor, out: &mut Tensor) -> Result<(), SmeltError> {
    out.cuda().dtod_copy(weights.data(), out.data_mut())?;
    Ok(())
}

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
    c.cuda().memset_zeros(c.data_mut())?;

    let batching: usize = a.shape()[..dim - 2].iter().product();
    let a_skip: usize = m * k;
    let b_skip: usize = n * k;
    let c_skip: usize = m * n;

    let blas = a.blas();

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

    let dev = a.cuda();

    let module_name = "add_fwd_f32";
    if !dev.has_func(module_name, module_name) {
        dev.load_ptx(ADD_PTX.into(), module_name, &[module_name])?;
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

    let dev = a.cuda();

    let module_name = "badd_fwd_f32";
    if !dev.has_func(module_name, module_name) {
        dev.load_ptx(ADD_PTX.into(), module_name, &[module_name])?;
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

    let dev = a.cuda();

    let module_name = "mul_fwd_f32";
    if !dev.has_func(module_name, module_name) {
        dev.load_ptx(ADD_PTX.into(), module_name, &[module_name])?;
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

    let dev = a.cuda();

    let module_name = "bmul_fwd_f32";
    if !dev.has_func(module_name, module_name) {
        dev.load_ptx(ADD_PTX.into(), module_name, &[module_name])?;
    }

    let numel = b.data().len();

    let fwd_fn = dev.get_func(module_name, module_name).unwrap();
    let cfg = LaunchConfig::for_num_elems(numel as u32);
    let params = (numel, a.data(), b.data_mut(), skip);
    unsafe { fwd_fn.launch(cfg, params) }?;

    Ok(())
}

const NORMALIZE_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/normalize.ptx"));

/// Basic operation for the layernorm.
/// x = (x - x.mean()) / (x.var() + epsilon)
/// `mean` and `var` do not have to be initialized, they are simply passed to
/// avoid allocation.
pub fn normalize(x: &mut Tensor, epsilon: f32) -> Result<(), SmeltError> {
    let dim = x.shape().len();
    let numel: usize = x.shape()[..dim - 1].iter().product();
    let size = x.shape()[dim - 1];
    let dev = x.cuda();

    let module_name = "normalize_f32";
    if !dev.has_func(module_name, module_name) {
        dev.load_ptx(NORMALIZE_PTX.into(), module_name, &[module_name])?;
    }

    let fwd_fn = dev.get_func(module_name, module_name).unwrap();
    let cfg = LaunchConfig::for_num_elems(numel as u32);
    let params = (numel, x.data_mut(), size, epsilon);
    unsafe { fwd_fn.launch(cfg, params) }?;

    Ok(())
}

const SOFTMAX_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/softmax.ptx"));

#[inline]
fn g_softmax<const CAUSAL: bool>(
    x: &mut Tensor,
    past_sequence_length: usize,
) -> Result<(), SmeltError> {
    let dim = x.shape().len();

    let m = x.shape()[dim - 2];
    let n = x.shape()[dim - 1];

    let dev = x.cuda();

    let module_name = "softmax_f32";
    if !dev.has_func(module_name, module_name) {
        dev.load_ptx(SOFTMAX_PTX.into(), module_name, &[module_name])?;
    }
    let past_sequence_length = if CAUSAL { past_sequence_length } else { n };

    let numel: usize = x.shape()[..dim - 1].iter().product();
    let fwd_fn = dev.get_func(module_name, module_name).unwrap();
    let cfg = LaunchConfig::for_num_elems(numel as u32);
    let params = (numel, x.data_mut(), m, n, past_sequence_length);
    unsafe { fwd_fn.launch(cfg, params) }?;

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

const UNITARY_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/unitary.ptx"));
/// utility function to use a faster but less precise tanh
#[inline]
pub fn tanh(x: &mut Tensor) -> Result<(), SmeltError> {
    let dev = x.cuda();
    let module_name = "tanh_f32";
    if !dev.has_func(module_name, module_name) {
        dev.load_ptx(UNITARY_PTX.into(), module_name, &[module_name])?;
    }
    let numel: usize = x.shape().iter().product();
    let fwd_fn = dev.get_func(module_name, module_name).unwrap();
    let cfg = LaunchConfig::for_num_elems(numel as u32);
    let params = (numel, x.data_mut());
    unsafe { fwd_fn.launch(cfg, params) }?;

    Ok(())
}

/// `gelu` operation
/// <https://en.wikipedia.org/wiki/Activation_function#Comparison_of_activation_functions>
/// but using [faster_tanh]
#[inline]
pub fn gelu(x: &mut Tensor) -> Result<(), SmeltError> {
    let module_name = "gelu_f32";
    let dev = x.cuda();
    if !dev.has_func(module_name, module_name) {
        dev.load_ptx(UNITARY_PTX.into(), module_name, &[module_name])?;
    }
    let numel: usize = x.shape().iter().product();
    let fwd_fn = dev.get_func(module_name, module_name).unwrap();
    let cfg = LaunchConfig::for_num_elems(numel as u32);
    let params = (numel, x.data_mut());
    unsafe { fwd_fn.launch(cfg, params) }?;
    Ok(())
}

/// TODO
#[inline]
pub fn mul_scalar(x: &mut Tensor, factor: f32) -> Result<(), SmeltError> {
    let dev = x.cuda();
    let module_name = "mul_scalar_f32";
    if !dev.has_func(module_name, module_name) {
        dev.load_ptx(UNITARY_PTX.into(), module_name, &[module_name])?;
    }
    let numel: usize = x.shape().iter().product();
    let fwd_fn = dev.get_func(module_name, module_name).unwrap();
    let cfg = LaunchConfig::for_num_elems(numel as u32);
    let params = (numel, x.data_mut(), factor);
    unsafe { fwd_fn.launch(cfg, params) }?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::f32::Device;
    use crate::tests::simplify;

    fn device() -> Device {
        Device::new(0).unwrap()
    }

    #[test]
    fn simple_matmul() {
        let device = device();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let a = Tensor::from_cpu(&data, vec![2, 2], &device).unwrap();
        let b = Tensor::from_cpu(&data, vec![2, 2], &device).unwrap();
        let mut c = Tensor::zeros(vec![2, 2], &device).unwrap();

        matmul(&a, &b, &mut c).unwrap();
        assert_eq!(c.cpu_data().unwrap(), &[7.0, 10.0, 15.0, 22.0]);
        matmul(&a, &b, &mut c).unwrap();
        assert_eq!(c.cpu_data().unwrap(), &[7.0, 10.0, 15.0, 22.0]);

        let data = vec![1.0, 2.0, 3.0];
        let a = Tensor::from_cpu(&data, vec![3, 1], &device).unwrap();
        let b = Tensor::from_cpu(&vec![3.0, 4.0], vec![1, 2], &device).unwrap();
        let mut c = Tensor::zeros(vec![3, 2], &device).unwrap();
        matmul(&a, &b, &mut c).unwrap();
        assert_eq!(c.cpu_data().unwrap(), &[3.0, 4.0, 6.0, 8.0, 9.0, 12.0]);

        let data: Vec<_> = (0..6).map(|i| i as f32).collect();
        let a = Tensor::from_cpu(&data, vec![2, 3], &device).unwrap();
        let data: Vec<_> = (0..12).map(|i| (i + 2) as f32).collect();
        let b = Tensor::from_cpu(&data, vec![3, 4], &device).unwrap();
        let mut c = Tensor::zeros(vec![2, 4], &device).unwrap();
        matmul(&a, &b, &mut c).unwrap();
        assert_eq!(
            c.cpu_data().unwrap(),
            &[26., 29., 32., 35., 80., 92., 104., 116.]
        );

        let data: Vec<_> = (0..12).map(|i| i as f32).collect();
        let a = Tensor::from_cpu(&data, vec![2, 2, 3], &device).unwrap();
        let data: Vec<_> = (0..24).map(|i| (i + 2) as f32).collect();
        let b = Tensor::from_cpu(&data, vec![2, 3, 4], &device).unwrap();
        let mut c: Tensor = Tensor::zeros(vec![2, 2, 4], &device).unwrap();
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
        let device = device();
        let a = Tensor::from_cpu(&vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], &device).unwrap();
        // A.T
        let b = Tensor::from_cpu(&vec![1.0, 3.0, 2.0, 4.0], vec![2, 2], &device).unwrap();
        let mut c = Tensor::zeros(vec![2, 2], &device).unwrap();

        matmul_t(&a, &b, &mut c).unwrap();
        assert_eq!(c.cpu_data().unwrap(), &[7.0, 10.0, 15.0, 22.0]);

        let a = Tensor::from_cpu(&vec![1.0, 2.0], vec![2, 1], &device).unwrap();
        let b = Tensor::from_cpu(&vec![3.0, 4.0], vec![2, 1], &device).unwrap();
        let mut c = Tensor::zeros(vec![2, 2], &device).unwrap();
        matmul_t(&a, &b, &mut c).unwrap();
        assert_eq!(c.cpu_data().unwrap(), &[3.0, 4.0, 6.0, 8.0]);

        let data: Vec<_> = (0..6).map(|i| i as f32).collect();
        let a = Tensor::from_cpu(&data, vec![2, 3], &device).unwrap();
        let data: Vec<_> = (0..12).map(|i| (i + 2) as f32).collect();
        let b = Tensor::from_cpu(&data, vec![4, 3], &device).unwrap();
        let mut c = Tensor::zeros(vec![2, 4], &device).unwrap();
        matmul_t(&a, &b, &mut c).unwrap();
        assert_eq!(
            c.cpu_data().unwrap(),
            &[11., 20., 29., 38., 38., 74., 110., 146.]
        );

        let data: Vec<_> = (0..12).map(|i| i as f32).collect();
        let a = Tensor::from_cpu(&data, vec![2, 2, 3], &device).unwrap();
        let data: Vec<_> = (0..24).map(|i| (i + 2) as f32).collect();
        let b = Tensor::from_cpu(&data, vec![2, 4, 3], &device).unwrap();
        let mut c = Tensor::zeros(vec![2, 2, 4], &device).unwrap();
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
        let device = device();
        let a = Tensor::from_cpu(&vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], &device).unwrap();
        let mut b = Tensor::from_cpu(&vec![1.0, 1.0, 1.0, 1.0], vec![2, 2], &device).unwrap();
        add(&a, &mut b).unwrap();
        assert_eq!(
            b.cpu_data().unwrap(),
            // Values obtained through python
            [2.0, 3.0, 4.0, 5.0]
        );
    }

    #[test]
    fn simple_broadcast_add() {
        let device = device();
        let a = Tensor::from_cpu(&vec![1.0, 2.0], vec![2], &device).unwrap();
        let mut b =
            Tensor::from_cpu(&vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0], vec![3, 2], &device).unwrap();
        broadcast_add(&a, &mut b).unwrap();
        assert_eq!(
            b.cpu_data().unwrap(),
            // Values obtained through python
            [2.0, 3.0, 2.0, 3.0, 2.0, 3.0]
        );
    }

    #[test]
    fn simple_mul() {
        let device = device();
        let a = Tensor::from_cpu(&vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], &device).unwrap();
        let mut b = Tensor::from_cpu(&vec![2.0, 2.0, 3.0, 3.0], vec![2, 2], &device).unwrap();
        mul(&a, &mut b).unwrap();
        assert_eq!(
            b.cpu_data().unwrap(),
            // Values obtained through python
            [2.0, 4.0, 9.0, 12.0]
        );
    }

    #[test]
    fn simple_softmax() {
        let device = device();
        let mut a = Tensor::from_cpu(&vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], &device).unwrap();
        softmax(&mut a).unwrap();
        assert_eq!(
            simplify(&a.cpu_data().unwrap()),
            // Values obtained through python
            [0.2689, 0.7311, 0.2689, 0.7311]
        );
    }

    #[test]
    fn simple_causal_softmax() {
        let device = device();
        let mut a = Tensor::from_cpu(&vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], &device).unwrap();
        // Large enough for the second test
        causal_softmax(&mut a, 0).unwrap();
        assert_eq!(
            simplify(&a.cpu_data().unwrap()),
            // Values obtained through python
            [1.0000, 0.0000, 0.2689, 0.7311]
        );

        let mut a = Tensor::from_cpu(&vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], &device).unwrap();
        causal_softmax(&mut a, 1).unwrap();
        assert_eq!(
            simplify(&a.cpu_data().unwrap()),
            // Values obtained through python
            [0.2689, 0.7311, 0.2689, 0.7311]
        );

        let data: Vec<_> = (0..12).map(|i| (i + 1) as f32).collect();
        let mut a = Tensor::from_cpu(&data, vec![3, 2, 2], &device).unwrap();
        causal_softmax(&mut a, 0).unwrap();
        assert_eq!(
            simplify(&a.cpu_data().unwrap()),
            // Values obtained through python
            [
                1.0000, 0.0000, 0.2689, 0.7311, 1.0000, 0.0000, 0.2689, 0.7311, 1.0000, 0.0000,
                0.2689, 0.7311
            ]
        );

        let data: Vec<_> = (0..12).map(|i| (i + 1) as f32).collect();
        let mut a = Tensor::from_cpu(&data, vec![2, 2, 3], &device).unwrap();
        causal_softmax(&mut a, 1).unwrap();
        assert_eq!(
            simplify(&a.cpu_data().unwrap()),
            // Values obtained through python
            [
                0.2689, 0.7311, 0.0, 0.09, 0.2447, 0.6652, 0.2689, 0.7311, 0.0, 0.09, 0.2447,
                0.6652
            ]
        );
    }

    #[test]
    fn simple_select() {
        let device = device();
        let a = Tensor::from_cpu(&vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], &device).unwrap();
        let mut tensor = Tensor::zeros(vec![3, 2], &device).unwrap();
        select(&[1, 0, 0], &a, &mut tensor).unwrap();
        assert_eq!(
            simplify(&tensor.cpu_data().unwrap()),
            // Values obtained through python
            [3.0, 4.0, 1.0, 2.0, 1.0, 2.0]
        );
    }

    #[test]
    fn simple_normalize() {
        let device = device();
        let mut a = Tensor::from_cpu(&vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], &device).unwrap();
        let epsilon = 1e-5;
        normalize(&mut a, epsilon).unwrap();
        assert_eq!(
            simplify(&a.cpu_data().unwrap()),
            // Values obtained through python
            [-1.0, 1.0, -1.0, 1.0]
        );

        // TODO Figure out how torch does layernorm to make sure this works.
        // let mut a = Tensor::from_cpu(&[-0.8570, -1.4722, -1.7398, -0.5307, -0.4816,  0.2071], vec![2, 3], 0).unwrap();
        // let epsilon = 1e-5;
        // normalize(&mut a, epsilon).unwrap();
        // assert_eq!(
        //     simplify(&a.cpu_data().unwrap()),
        //     // Values obtained through python
        //     [ 1.1031, -0.2559, -0.8472, -0.6359, -0.5167,  1.1526]
        // );
    }
}
