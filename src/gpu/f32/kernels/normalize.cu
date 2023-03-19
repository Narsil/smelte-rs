extern "C" __global__ void normalize_f32( 
    const size_t numel, 
    float *lhs, 
    const size_t size, 
    const float epsilon
) { 
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i >= numel) { 
        return; 
    } 

    const size_t offset = i * size;

    float sum = 0.0;
    for (int i=0; i < size; i ++){
	sum += lhs[offset + i];
    }
    sum /= size;
    for (int i=0; i < size; i ++){
	lhs[offset + i] -= sum;
    }

    float var = 0.0;
    for (int i=0; i < size; i ++){
	var += lhs[offset + i] * lhs[offset + i];
    }
    var /= size;
    var += epsilon;
    const float std = sqrt(var);
    for (int i=0; i < size; i ++){
	lhs[offset + i] /= std;
    }
} 

