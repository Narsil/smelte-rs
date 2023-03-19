#include "cuda_utils.cuh"

#define LONG_OP(TYPENAME, FORWARD, FUNC) \
extern "C" __global__ void FORWARD( \
    const size_t numel, \
    const TYPENAME *lhs, \
    TYPENAME *rhs \
) { \
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= numel) { \
        return; \
    } \
\
    TYPENAME x = lhs[i]; \
    TYPENAME y = rhs[i]; \
    TYPENAME fx; \
\
    FUNC\
\
    rhs[i] = fx; \
} \

#define OP(TYPENAME, FORWARD, FUNC) \
    LONG_OP(TYPENAME, FORWARD, fx = (FUNC);)

#define LONG_BROADCAST_OP(TYPENAME, FORWARD, FUNC) \
extern "C" __global__ void FORWARD( \
    const size_t numel, \
    const TYPENAME *lhs, \
    TYPENAME *rhs, \
    const size_t skip \
) { \
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= numel) { \
        return; \
    } \
\
    TYPENAME x = lhs[i % skip]; \
    TYPENAME y = rhs[i]; \
    TYPENAME fx; \
\
    FUNC\
\
    rhs[i] = fx; \
} \

#define BROADCAST_OP(TYPENAME, FORWARD, FUNC) \
    LONG_BROADCAST_OP(TYPENAME, FORWARD, fx = (FUNC);)
