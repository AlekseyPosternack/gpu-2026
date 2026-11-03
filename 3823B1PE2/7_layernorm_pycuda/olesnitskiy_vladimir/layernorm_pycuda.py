import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

_BLOCK_SIZE = 256

_KERNEL_SOURCE = r"""
#define BLOCK_SIZE %(block_size)d
#define WARP_SIZE 32

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, offset);
    }
    return v;
}

__device__ float block_sum(float v, float* shared) {
    const int lane = threadIdx.x %% WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    v = warp_sum(v);
    if (lane == 0) {
        shared[warp] = v;
    }
    __syncthreads();
    if (warp == 0) {
        v = lane < BLOCK_SIZE / WARP_SIZE ? shared[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) {
            shared[0] = v;
        }
    }
    __syncthreads();
    const float result = shared[0];
    __syncthreads();
    return result;
}

extern "C" __global__ void layernorm(const float* __restrict__ input,
                                     const float* __restrict__ gamma,
                                     const float* __restrict__ beta,
                                     float* __restrict__ output,
                                     int row_size, float eps) {
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];
    const float* src = input + (size_t)blockIdx.x * row_size;
    float* dst = output + (size_t)blockIdx.x * row_size;
    const float inv_n = 1.0f / row_size;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += BLOCK_SIZE) {
        sum += src[i];
    }
    const float mean = block_sum(sum, shared) * inv_n;

    float sq_sum = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += BLOCK_SIZE) {
        const float d = src[i] - mean;
        sq_sum += d * d;
    }
    const float inv_std = rsqrtf(block_sum(sq_sum, shared) * inv_n + eps);

    for (int i = threadIdx.x; i < row_size; i += BLOCK_SIZE) {
        dst[i] = (src[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}
""" % {"block_size": _BLOCK_SIZE}

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        module = SourceModule(_KERNEL_SOURCE, no_extern_c=True,
                              options=["-O3", "--use_fast_math"])
        _kernel = module.get_function("layernorm")
    return _kernel


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    """
    Apply Layer Normalization to each row of the input matrix.

    Parameters
    ----------
    input : list or numpy.ndarray of float
        Flattened matrix in row‑major order. Its length must be divisible by row_size.
    gamma : list or numpy.ndarray of float
        Scale parameter, length = row_size.
    beta : list or numpy.ndarray of float
        Shift parameter, length = row_size.
    row_size : int
        Number of features per row (i.e., number of columns).
    eps : float, optional
        Small constant for numerical stability.

    Returns
    -------
    numpy.ndarray
        Flattened matrix of the same shape as input, containing the row‑wise
        normalized results.
    """
    x = np.ascontiguousarray(input, dtype=np.float32).ravel()
    g = np.ascontiguousarray(gamma, dtype=np.float32).ravel()
    b = np.ascontiguousarray(beta, dtype=np.float32).ravel()
    row_size = int(row_size)
    if x.size == 0:
        return np.empty(0, dtype=np.float32)
    row_count = x.size // row_size

    output = np.empty_like(x)
    _get_kernel()(cuda.In(x), cuda.In(g), cuda.In(b), cuda.Out(output),
                  np.int32(row_size), np.float32(eps),
                  block=(_BLOCK_SIZE, 1, 1), grid=(row_count, 1))
    return output
