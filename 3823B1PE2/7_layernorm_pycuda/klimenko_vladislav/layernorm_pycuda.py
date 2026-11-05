import math
import numpy as np
from numba import cuda, float32

THREADS_PER_BLOCK = 256

@cuda.jit(device=True)
def _warp_reduce_sum(val):
    for offset in (16, 8, 4, 2, 1):
        val += cuda.shfl_down_sync(0xffffffff, val, offset)
    return val

@cuda.jit(device=True)
def _block_reduce_sum(val, shared):
    lane = cuda.threadIdx.x & 31
    wid  = cuda.threadIdx.x >> 5
    nwarps = cuda.blockDim.x >> 5
    val = _warp_reduce_sum(val)
    if lane == 0:
        shared[wid] = val
    cuda.syncthreads()
    if wid == 0:
        v = shared[lane] if lane < nwarps else float32(0)
        v = _warp_reduce_sum(v)
        if lane == 0:
            shared[0] = v
    cuda.syncthreads()
    return shared[0]

@cuda.jit
def _layernorm_kernel(values, gamma, beta, row_size, eps):
    row = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    start = row * row_size
    shared = cuda.shared.array(32, float32)

    ref = values[start]

    diff_sum = float32(0)
    for i in range(tid, row_size, cuda.blockDim.x):
        diff_sum += values[start + i] - ref
    mean_diff = _block_reduce_sum(diff_sum, shared) / float32(row_size)

    sq_sum = float32(0)
    for i in range(tid, row_size, cuda.blockDim.x):
        c = values[start + i] - ref - mean_diff
        sq_sum += c * c
    var = _block_reduce_sum(sq_sum, shared) / float32(row_size)
    inv_std = float32(1) / math.sqrt(var + eps)

    for i in range(tid, row_size, cuda.blockDim.x):
        c = values[start + i] - ref - mean_diff
        values[start + i] = c * inv_std * gamma[i] + beta[i]


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    x = np.ascontiguousarray(input, dtype=np.float32)
    g = np.ascontiguousarray(gamma, dtype=np.float32)
    b = np.ascontiguousarray(beta, dtype=np.float32)
    d_x = cuda.to_device(x)
    d_g = cuda.to_device(g)
    d_b = cuda.to_device(b)
    row_count = x.size // row_size
    _layernorm_kernel[row_count, THREADS_PER_BLOCK](d_x, d_g, d_b, row_size, np.float32(eps))
    return d_x.copy_to_host()