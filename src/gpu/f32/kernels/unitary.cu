
__device__ __forceinline__ float tanhg(float a) { return tanhf(a); }
__device__ __forceinline__ double tanhg(double a) { return tanh(a); }

template<typename T>
__device__ T gelu_fwd(T x) {
    constexpr T fastCoeff = 0.044715;
    T x_sq = x * x;
    T x_cube = x_sq * x;
    T alpha = x + fastCoeff * x_cube;
    return 0.5 * x * (1.0 + tanhg(M_2_SQRTPI * M_SQRT1_2 * alpha));
}


extern "C" __global__ void tanh_f32( 
    const size_t numel, 
    float *x
) { 
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i >= numel) { 
        return; 
    } 
    x[i] = tanhf(x[i]);
} 

extern "C" __global__ void gelu_f32( 
    const size_t numel, 
    float *x 
) { 
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i >= numel) { 
        return; 
    } 
    x[i] = gelu_fwd(x[i]);
} 

extern "C" __global__ void mul_scalar_f32( 
    const size_t numel, 
    float *x ,
    float factor
) { 
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i >= numel) { 
        return; 
    } 
    x[i] *= factor;
} 
