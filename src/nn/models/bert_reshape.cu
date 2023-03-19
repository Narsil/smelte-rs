extern "C" __global__ void split_heads(
    const size_t numel,
    const float *q,
    float *q_split,
    const size_t num_heads,
    const size_t sequence_length,
    const size_t head_dim
) {
    size_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= numel) {
        return;
    }

    const size_t k = n % head_dim;
    const size_t j = (n / head_dim) % sequence_length;
    const size_t i = n / head_dim / sequence_length;

    const size_t hidden_dim = num_heads * head_dim;
    const size_t index = j * hidden_dim + i * head_dim + k;
    const size_t out_index = i * sequence_length * head_dim + j * head_dim + k;
    q_split[out_index] = q[index];
}

extern "C" __global__ void unsplit_heads(
    const size_t numel,
    const float *q_split,
    float *q,
    const size_t num_heads,
    const size_t sequence_length,
    const size_t head_dim
) {
    size_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= numel) {
        return;
    }

    const size_t k = n % head_dim;
    const size_t j = (n / head_dim) % sequence_length;
    const size_t i = n / head_dim / sequence_length;

    const size_t hidden_dim = num_heads * head_dim;
    const size_t in_index = i * sequence_length * head_dim + j * head_dim + k;
    const size_t out_index = j * hidden_dim + i * head_dim + k;

    q[out_index] = q_split[in_index];
}
