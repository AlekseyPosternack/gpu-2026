import numpy as np

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

_KERNEL_SRC = r"""
#define WARP_SIZE 32
#define MAX_WARPS 8

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float block_reduce_sum(float v, float* s_warp,
                                                  int tid, int num_warps) {
    const int warp = tid >> 5, lane = tid & 31;
    __syncthreads();
    v = warp_reduce_sum(v);
    if (lane == 0) s_warp[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = (lane < num_warps) ? s_warp[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) s_warp[0] = v;
    }
    __syncthreads();
    return s_warp[0];
}

__global__ void layernorm_kernel(const float* __restrict__ in,
                                 const float* __restrict__ gamma,
                                 const float* __restrict__ beta,
                                 float* __restrict__ out,
                                 int row_size, float eps, int cache_row)
{
    extern __shared__ float s_row[];
    __shared__ float s_warp[MAX_WARPS];

    const int tid      = threadIdx.x;
    const int nthreads = blockDim.x;
    const long base    = (long)blockIdx.x * row_size;
    const float* row_in  = in  + base;
    float*       row_out = out + base;

    if (cache_row) {
        for (int i = tid; i < row_size; i += nthreads)
            s_row[i] = row_in[i];
        __syncthreads();
    }

    float local = 0.0f;
    for (int i = tid; i < row_size; i += nthreads)
        local += cache_row ? s_row[i] : row_in[i];
    const float mean = block_reduce_sum(local, s_warp, tid, nthreads >> 5)
                       / (float)row_size;

    local = 0.0f;
    for (int i = tid; i < row_size; i += nthreads) {
        float d = (cache_row ? s_row[i] : row_in[i]) - mean;
        local += d * d;
    }
    const float var  = block_reduce_sum(local, s_warp, tid, nthreads >> 5)
                       / (float)row_size;
    const float rstd = rsqrtf(var + eps);    // 1 / sqrt(var + eps)

    for (int i = tid; i < row_size; i += nthreads) {
        float x = cache_row ? s_row[i] : row_in[i];
        row_out[i] = gamma[i] * ((x - mean) * rstd) + beta[i];
    }
}
"""

_SHARED_CAP = 48 * 1024 // 4

_state = {
    "func": None, "stream": None,
    "d_in": None, "d_out": None, "d_gamma": None, "d_beta": None,
    "total": 0, "row": 0,
}


def _get_func():
    if _state["func"] is None:
        mod = SourceModule(_KERNEL_SRC, options=["--use_fast_math"])
        _state["func"] = mod.get_function("layernorm_kernel")
        _state["stream"] = cuda.Stream()
    return _state["func"]


def _ensure_buffers(total, row_size):
    if total > _state["total"]:
        _state["d_in"]  = cuda.mem_alloc(total * 4)
        _state["d_out"] = cuda.mem_alloc(total * 4)
        _state["total"] = total
    if row_size > _state["row"]:
        _state["d_gamma"] = cuda.mem_alloc(row_size * 4)
        _state["d_beta"]  = cuda.mem_alloc(row_size * 4)
        _state["row"] = row_size


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    x = np.ascontiguousarray(input, dtype=np.float32).ravel()
    g = np.ascontiguousarray(gamma, dtype=np.float32).ravel()
    b = np.ascontiguousarray(beta,  dtype=np.float32).ravel()

    total     = x.size
    row_count = total // row_size

    func   = _get_func()
    _ensure_buffers(total, row_size)
    stream = _state["stream"]

    cuda.memcpy_htod_async(_state["d_in"],    x, stream)
    cuda.memcpy_htod_async(_state["d_gamma"], g, stream)
    cuda.memcpy_htod_async(_state["d_beta"],  b, stream)

    threads = min(256, max(32, row_size))
    cache   = 1 if row_size <= _SHARED_CAP else 0
    shared  = row_size * 4 if cache else 0

    func(_state["d_in"], _state["d_gamma"], _state["d_beta"], _state["d_out"],
         np.int32(row_size), np.float32(eps), np.int32(cache),
         block=(threads, 1, 1), grid=(row_count, 1, 1),
         shared=shared, stream=stream)

    out = np.empty(total, dtype=np.float32)
    cuda.memcpy_dtoh_async(out, _state["d_out"], stream)
    stream.synchronize()
    return out
