extern "C" __global__ void softmax_f32( 
    const size_t numel, 
    float *lhs, 
    const size_t m, 
    const size_t size, 
    const size_t past_sequence_length
) { 
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i >= numel) { 
        return; 
    } 

    const size_t offset = i * size;
    i = i % m;
    float current_max = -1 * INFINITY;
    for (int j = 0; j< size; j++){
	    const float v = lhs[offset + j];
	    if (v > current_max && i + past_sequence_length >=j) {
		    current_max = v;
	    }
    }

    for (int j = 0; j< size; j++){
	    lhs[offset + j] -= current_max;
	    lhs[offset + j] = exp(lhs[offset + j]);
    }

    float sum = 0.0;
    for (int j = 0; j< size; j++){
	    const float v = lhs[offset + j];
	    if (i + past_sequence_length >=j){
		    sum += v;
	    }
    }

    for (int j = 0; j< size; j++){
	    if (i + past_sequence_length >=j){
		    lhs[offset + j] /= sum;
	    }else{
		    lhs[offset + j] = 0.0;
	    }
    }

} 

