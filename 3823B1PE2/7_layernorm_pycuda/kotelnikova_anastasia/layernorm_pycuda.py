import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

_KERNEL_SRC = r'''
#include <math.h>

#define THREADS 256
#define NWARPS  (THREADS / 32)

__device__ __forceinline__ float warpReduceSum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

__global__ void layernorm_kernel(const float* __restrict__ in,
                                 const float* __restrict__ gamma,
                                 const float* __restrict__ beta,
                                 float* __restrict__ out,
                                 int row_size,
                                 float eps)
{
    __shared__ float s_warp[NWARPS];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;

    const float* row_in  = in  + (size_t)row * row_size;
    float*       row_out = out + (size_t)row * row_size;

    float local_sum = 0.0f;
    for (int i = tid; i < row_size; i += THREADS)
        local_sum += row_in[i];

    local_sum = warpReduceSum(local_sum);
    if (lane == 0) s_warp[warp] = local_sum;
    __syncthreads();

    float sum = s_warp[0];
    #pragma unroll
    for (int i = 1; i < NWARPS; ++i) sum += s_warp[i];
    const float mean = sum / (float)row_size;

    float local_var = 0.0f;
    for (int i = tid; i < row_size; i += THREADS) {
        const float d = row_in[i] - mean;
        local_var += d * d;
    }
    local_var = warpReduceSum(local_var);
    if (lane == 0) s_warp[warp] = local_var;
    __syncthreads();

    float var = s_warp[0];
    #pragma unroll
    for (int i = 1; i < NWARPS; ++i) var += s_warp[i];
    const float inv_std = rsqrtf(var / (float)row_size + eps);

    for (int i = tid; i < row_size; i += THREADS)
        row_out[i] = gamma[i] * (row_in[i] - mean) * inv_std + beta[i];
}
'''

_module = SourceModule(_KERNEL_SRC)
_kernel = _module.get_function("layernorm_kernel")

_gpu_cache = {
    'in': None, 'out': None, 'gamma': None, 'beta': None,
    'capacity': 0, 'row_size': 0,
}


def _ensure_buffers(total, row_size):
    global _gpu_cache
    if _gpu_cache['capacity'] >= total and _gpu_cache['row_size'] == row_size:
        return _gpu_cache

    for key in ('in', 'out', 'gamma', 'beta'):
        if _gpu_cache[key] is not None:
            _gpu_cache[key].free()

    _gpu_cache['in']    = cuda.mem_alloc(total * 4)
    _gpu_cache['out']   = cuda.mem_alloc(total * 4)
    _gpu_cache['gamma'] = cuda.mem_alloc(row_size * 4)
    _gpu_cache['beta']  = cuda.mem_alloc(row_size * 4)
    _gpu_cache['capacity'] = total
    _gpu_cache['row_size'] = row_size
    return _gpu_cache


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    x = np.asarray(input, dtype=np.float32).ravel()
    g = np.asarray(gamma, dtype=np.float32).ravel()
    b = np.asarray(beta,  dtype=np.float32).ravel()

    total = x.size
    if total == 0 or row_size <= 0 or total % row_size != 0:
        raise ValueError("input size must be positive and divisible by row_size")
    if g.size != row_size or b.size != row_size:
        raise ValueError("gamma and beta must have length == row_size")

    row_count = total // row_size
    out = np.empty_like(x)

    buf = _ensure_buffers(total, row_size)

    cuda.memcpy_htod(buf['in'],    x)
    cuda.memcpy_htod(buf['gamma'], g)
    cuda.memcpy_htod(buf['beta'],  b)

    _kernel(
        buf['in'], buf['gamma'], buf['beta'], buf['out'],
        np.int32(row_size),
        np.float32(eps),
        block=(256, 1, 1),
        grid=(row_count, 1, 1),
    )

    cuda.memcpy_dtoh(out, buf['out'])
    return out