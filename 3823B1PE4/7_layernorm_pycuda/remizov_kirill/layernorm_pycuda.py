import numpy as np
import pycuda.autoinit  # noqa: F401  (creates the CUDA context)
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


# One CUDA block processes one row of the input matrix. The row is read twice:
# once to compute the mean, once to compute the variance. The second read is
# usually served from L2 cache because a row is only row_size * 4 bytes.
# Vectorized float4 accesses keep the memory subsystem busy, and warp shuffles
# make the block-wide reductions cheap (no atomics, no extra global memory).
_KERNEL_SOURCE = r"""
#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_sum(float value) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

// Sums value across the whole block. Every thread receives the result.
// An extra barrier makes it safe to reuse the shared buffer after the call.
__device__ __forceinline__ float block_reduce_sum(float value, float* shared) {
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x / WARP_SIZE;

    value = warp_reduce_sum(value);
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();

    if (warp == 0) {
        value = lane < (blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;
        value = warp_reduce_sum(value);
        if (lane == 0) {
            shared[0] = value;
        }
    }
    __syncthreads();

    const float result = shared[0];
    __syncthreads();
    return result;
}

extern "C" __global__ void layernorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int row_size,
    float eps)
{
    __shared__ float shared[WARP_SIZE];

    const int tid = threadIdx.x;
    const size_t base = (size_t)blockIdx.x * (size_t)row_size;

    float mean;
    float inv_std;

    if ((row_size & 3) == 0) {
        // Fast path: row_size is a multiple of 4, so float4 loads are valid.
        const int vec_count = row_size >> 2;
        const float4* in4 = reinterpret_cast<const float4*>(input + base);
        float4* out4 = reinterpret_cast<float4*>(output + base);
        const float4* gamma4 = reinterpret_cast<const float4*>(gamma);
        const float4* beta4 = reinterpret_cast<const float4*>(beta);

        // Mean
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        for (int i = tid; i < vec_count; i += blockDim.x) {
            const float4 v = in4[i];
            s0 += v.x;
            s1 += v.y;
            s2 += v.z;
            s3 += v.w;
        }
        mean = block_reduce_sum((s0 + s1) + (s2 + s3), shared) /
               (float)row_size;

        // Variance (two-pass formula is numerically more stable than E[x^2] - E[x]^2)
        float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
        for (int i = tid; i < vec_count; i += blockDim.x) {
            const float4 v = in4[i];
            const float d0 = v.x - mean;
            const float d1 = v.y - mean;
            const float d2 = v.z - mean;
            const float d3 = v.w - mean;
            v0 += d0 * d0;
            v1 += d1 * d1;
            v2 += d2 * d2;
            v3 += d3 * d3;
        }
        inv_std = rsqrtf(block_reduce_sum((v0 + v1) + (v2 + v3), shared) /
                         (float)row_size + eps);

        // Normalize, scale and shift
        for (int i = tid; i < vec_count; i += blockDim.x) {
            const float4 v = in4[i];
            const float4 g = gamma4[i];
            const float4 b = beta4[i];

            float4 o;
            o.x = (v.x - mean) * inv_std * g.x + b.x;
            o.y = (v.y - mean) * inv_std * g.y + b.y;
            o.z = (v.z - mean) * inv_std * g.z + b.z;
            o.w = (v.w - mean) * inv_std * g.w + b.w;
            out4[i] = o;
        }
    } else {
        // Fallback for rows whose size is not a multiple of 4.
        const float* row_in = input + base;
        float* row_out = output + base;

        float sum = 0.0f;
        for (int i = tid; i < row_size; i += blockDim.x) {
            sum += row_in[i];
        }
        mean = block_reduce_sum(sum, shared) / (float)row_size;

        float sq = 0.0f;
        for (int i = tid; i < row_size; i += blockDim.x) {
            const float d = row_in[i] - mean;
            sq += d * d;
        }
        inv_std = rsqrtf(block_reduce_sum(sq, shared) / (float)row_size + eps);

        for (int i = tid; i < row_size; i += blockDim.x) {
            row_out[i] = (row_in[i] - mean) * inv_std * gamma[i] + beta[i];
        }
    }
}
"""

_BLOCK_SIZE = 256

_module = SourceModule(_KERNEL_SOURCE, options=["-O3"])
_kernel = _module.get_function("layernorm_kernel")

# Device buffers are allocated once and reused between calls: the benchmark
# invokes the function several times with the same matrix size.
_BUFFERS = {}


def _device_buffer(name, nbytes):
    entry = _BUFFERS.get(name)
    if entry is None or entry[0] < nbytes:
        # The previous allocation is released automatically once it is
        # replaced in the dictionary.
        allocation = cuda.mem_alloc(max(nbytes, 1))
        _BUFFERS[name] = (nbytes, allocation)
        return allocation
    return entry[1]


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
    if row_size <= 0:
        raise ValueError("row_size must be positive")
    if x.size % row_size != 0:
        raise ValueError("input size must be divisible by row_size")
    if g.size != row_size or b.size != row_size:
        raise ValueError("gamma and beta must have row_size elements")

    out = np.empty_like(x)
    if x.size == 0:
        return out

    row_count = x.size // row_size

    d_in = _device_buffer("input", x.nbytes)
    d_out = _device_buffer("output", x.nbytes)
    d_gamma = _device_buffer("gamma", g.nbytes)
    d_beta = _device_buffer("beta", b.nbytes)

    cuda.memcpy_htod(d_in, x)
    cuda.memcpy_htod(d_gamma, g)
    cuda.memcpy_htod(d_beta, b)

    _kernel(
        d_in,
        d_gamma,
        d_beta,
        d_out,
        np.int32(row_size),
        np.float32(eps),
        block=(_BLOCK_SIZE, 1, 1),
        grid=(row_count, 1, 1),
    )

    cuda.memcpy_dtoh(out, d_out)
    return out
