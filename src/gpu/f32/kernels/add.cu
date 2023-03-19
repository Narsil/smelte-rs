#include "binary_op_macros.cuh"

OP(float, add_fwd_f32, x + y)
OP(float, mul_fwd_f32, x * y)
BROADCAST_OP(float, badd_fwd_f32, x + y)
BROADCAST_OP(float, bmul_fwd_f32, x * y)


