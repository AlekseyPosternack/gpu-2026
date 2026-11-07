import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

_KERNEL_SOURCE = r"""
__device__ float BlockReduceSum(float v) {
    __shared__ float scratch[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    if (lane == 0) scratch[warp] = v;
    __syncthreads();

    v = (lane < (blockDim.x >> 5)) ? scratch[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    __syncthreads();
    return v;
}

extern "C" __global__ void LayerNormKernel(const float* __restrict__ in,
                                           const float* __restrict__ gamma,
                                           const float* __restrict__ beta,
                                           float* __restrict__ out,
                                           int row_size, float eps, int cached) {
    extern __shared__ float row_cache[];

    const float* x = in + (size_t)blockIdx.x * row_size;
    float* y = out + (size_t)blockIdx.x * row_size;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        const float v = x[i];
        if (cached) row_cache[i] = v;
        sum += v;
    }
    const float mean = BlockReduceSum(sum) / row_size;

    float sq = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        const float v = cached ? row_cache[i] : x[i];
        const float d = v - mean;
        sq += d * d;
    }
    const float var = BlockReduceSum(sq) / row_size;
    const float inv_std = rsqrtf(var + eps);

    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        const float v = cached ? row_cache[i] : x[i];
        y[i] = (v - mean) * inv_std * gamma[i] + beta[i];
    }
}
"""

_MAX_THREADS = 256
_MAX_SHARED_BYTES = 48 * 1024

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = SourceModule(_KERNEL_SOURCE, no_extern_c=True).get_function("LayerNormKernel")
        _kernel.prepare("PPPPifi")
    return _kernel


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    x = np.ascontiguousarray(input, dtype=np.float32).ravel()
    g = np.ascontiguousarray(gamma, dtype=np.float32).ravel()
    b = np.ascontiguousarray(beta, dtype=np.float32).ravel()

    if x.size == 0:
        return np.empty(0, dtype=np.float32)

    row_size = int(row_size)
    row_count = x.size // row_size

    d_in = gpuarray.to_gpu(x)
    d_gamma = gpuarray.to_gpu(g)
    d_beta = gpuarray.to_gpu(b)
    d_out = gpuarray.empty_like(d_in)

    block = min(_MAX_THREADS, max(32, row_size))
    shared_bytes = row_size * 4
    cached = shared_bytes <= _MAX_SHARED_BYTES

    kernel = _get_kernel()
    kernel.prepared_call(
        (row_count, 1), (block, 1, 1),
        d_in.gpudata, d_gamma.gpudata, d_beta.gpudata, d_out.gpudata,
        np.int32(row_size), np.float32(eps), np.int32(1 if cached else 0),
        shared_size=shared_bytes if cached else 0,
    )

    return d_out.get()  