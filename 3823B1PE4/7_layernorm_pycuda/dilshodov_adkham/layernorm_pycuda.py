import numpy as np
import pycuda.autoinit  # noqa: F401  (creates CUDA context)
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

_KERNEL_SOURCE = r"""
#define WARP_SIZE 32
#define MAX_BLOCK_SIZE 256

__device__ __forceinline__ float WarpSum(float v) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, offset);
    return v;
}

// Sum over the whole block, result is returned to every thread
__device__ float BlockSum(float v, float* shared) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    v = WarpSum(v);
    if (lane == 0) shared[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = lane < blockDim.x / WARP_SIZE ? shared[lane] : 0.0f;
        v = WarpSum(v);
        if (lane == 0) shared[0] = v;
    }
    __syncthreads();
    const float result = shared[0];
    __syncthreads();  // shared buffer is reused by the next call
    return result;
}

// One CUDA block normalizes one row
__global__ void LayerNorm(const float* __restrict__ in, const float* __restrict__ gamma,
                          const float* __restrict__ beta, float* __restrict__ out,
                          int row_size, float eps) {
    __shared__ float shared[MAX_BLOCK_SIZE / WARP_SIZE];
    const float* row_in = in + (size_t)blockIdx.x * row_size;
    float* row_out = out + (size_t)blockIdx.x * row_size;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) sum += row_in[i];
    const float mean = BlockSum(sum, shared) / row_size;

    // Two-pass variance is more accurate than E[x^2] - E[x]^2
    float sq = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        const float d = row_in[i] - mean;
        sq += d * d;
    }
    const float inv_std = rsqrtf(BlockSum(sq, shared) / row_size + eps);

    for (int i = threadIdx.x; i < row_size; i += blockDim.x)
        row_out[i] = (row_in[i] - mean) * inv_std * gamma[i] + beta[i];
}
"""

_WARP_SIZE = 32
_MAX_BLOCK_SIZE = 256

# Kernel is compiled once on module import
_module = SourceModule(_KERNEL_SOURCE)
_kernel = _module.get_function("LayerNorm")

# Device buffers are allocated once and reused while sizes fit
_buffers = {}


def _device_buffer(name, nbytes):
    buf = _buffers.get(name)
    if buf is None or buf[1] < nbytes:
        buf = (cuda.mem_alloc(max(nbytes, 1)), nbytes)
        _buffers[name] = buf
    return buf[0]


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    """
    Apply Layer Normalization to each row of the input matrix.

    Parameters
    ----------
    input : list or numpy.ndarray of float
        Flattened matrix in row-major order. Its length must be divisible by row_size.
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
        Flattened matrix of the same shape as input, containing the row-wise
        normalized results.
    """
    x = np.ascontiguousarray(input, dtype=np.float32).ravel()
    g = np.ascontiguousarray(gamma, dtype=np.float32).ravel()
    b = np.ascontiguousarray(beta, dtype=np.float32).ravel()
    row_size = int(row_size)
    if x.size == 0:
        return np.empty(0, dtype=np.float32)
    row_count = x.size // row_size

    d_in = _device_buffer("in", x.nbytes)
    d_out = _device_buffer("out", x.nbytes)
    d_gamma = _device_buffer("gamma", g.nbytes)
    d_beta = _device_buffer("beta", b.nbytes)
    cuda.memcpy_htod(d_in, x)
    cuda.memcpy_htod(d_gamma, g)
    cuda.memcpy_htod(d_beta, b)

    block_size = _WARP_SIZE
    while block_size < _MAX_BLOCK_SIZE and block_size < row_size:
        block_size *= 2

    _kernel(d_in, d_gamma, d_beta, d_out, np.int32(row_size), np.float32(eps),
            block=(block_size, 1, 1), grid=(row_count, 1))

    out = np.empty_like(x)
    cuda.memcpy_dtoh(out, d_out)
    return out
