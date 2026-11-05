import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

_MAX_THREADS = 1024
_WARP_SIZE = 32

_KERNEL_SOURCE = r"""
#define WARP_SIZE 32

__device__ __forceinline__ float WarpSum(float value) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value += __shfl_xor_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ float BlockSum(float value, float* shared) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int warps = blockDim.x / WARP_SIZE;
    value = WarpSum(value);
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();
    value = WarpSum(lane < warps ? shared[lane] : 0.0f);
    __syncthreads();
    return value;
}

template <int kValuesPerThread>
__device__ void LayerNormCached(const float* __restrict__ input,
                                const float* __restrict__ gamma,
                                const float* __restrict__ beta,
                                float* __restrict__ output,
                                int row_size, float eps) {
    __shared__ float shared[WARP_SIZE];
    const size_t offset = (size_t)blockIdx.x * row_size;
    const float* in = input + offset;
    float* out = output + offset;
    const float inv_n = 1.0f / row_size;

    float values[kValuesPerThread];
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < kValuesPerThread; ++j) {
        const int i = threadIdx.x + j * blockDim.x;
        values[j] = i < row_size ? in[i] : 0.0f;
        sum += values[j];
    }
    const float mean = BlockSum(sum, shared) * inv_n;

    float sq_sum = 0.0f;
#pragma unroll
    for (int j = 0; j < kValuesPerThread; ++j) {
        const int i = threadIdx.x + j * blockDim.x;
        values[j] -= mean;
        if (i < row_size) {
            sq_sum += values[j] * values[j];
        }
    }
    const float inv_std = rsqrtf(BlockSum(sq_sum, shared) * inv_n + eps);

#pragma unroll
    for (int j = 0; j < kValuesPerThread; ++j) {
        const int i = threadIdx.x + j * blockDim.x;
        if (i < row_size) {
            out[i] = gamma[i] * (values[j] * inv_std) + beta[i];
        }
    }
}

extern "C" {

__global__ void layernorm_1(const float* input, const float* gamma, const float* beta,
                            float* output, int row_size, float eps) {
    LayerNormCached<1>(input, gamma, beta, output, row_size, eps);
}

__global__ void layernorm_2(const float* input, const float* gamma, const float* beta,
                            float* output, int row_size, float eps) {
    LayerNormCached<2>(input, gamma, beta, output, row_size, eps);
}

__global__ void layernorm_4(const float* input, const float* gamma, const float* beta,
                            float* output, int row_size, float eps) {
    LayerNormCached<4>(input, gamma, beta, output, row_size, eps);
}

__global__ void layernorm_8(const float* input, const float* gamma, const float* beta,
                            float* output, int row_size, float eps) {
    LayerNormCached<8>(input, gamma, beta, output, row_size, eps);
}

__global__ void layernorm_16(const float* input, const float* gamma, const float* beta,
                             float* output, int row_size, float eps) {
    LayerNormCached<16>(input, gamma, beta, output, row_size, eps);
}

__global__ void layernorm(const float* __restrict__ input,
                          const float* __restrict__ gamma,
                          const float* __restrict__ beta,
                          float* __restrict__ output,
                          int row_size, float eps) {
    __shared__ float shared[WARP_SIZE];
    const size_t offset = (size_t)blockIdx.x * row_size;
    const float* in = input + offset;
    float* out = output + offset;
    const float inv_n = 1.0f / row_size;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        sum += in[i];
    }
    const float mean = BlockSum(sum, shared) * inv_n;

    float sq_sum = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        const float d = in[i] - mean;
        sq_sum += d * d;
    }
    const float inv_std = rsqrtf(BlockSum(sq_sum, shared) * inv_n + eps);

    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        out[i] = gamma[i] * ((in[i] - mean) * inv_std) + beta[i];
    }
}

}
"""

_module = SourceModule(_KERNEL_SOURCE, no_extern_c=True)
_cached_kernels = {
    count: _module.get_function("layernorm_%d" % count) for count in (1, 2, 4, 8, 16)
}
_generic_kernel = _module.get_function("layernorm")


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
    if x.size == 0 or row_size <= 0:
        return np.empty(0, dtype=np.float32)
    row_count = x.size // row_size

    d_input = drv.mem_alloc(x.nbytes)
    d_output = drv.mem_alloc(x.nbytes)
    d_gamma = drv.mem_alloc(g.nbytes)
    d_beta = drv.mem_alloc(b.nbytes)
    drv.memcpy_htod(d_input, x)
    drv.memcpy_htod(d_gamma, g)
    drv.memcpy_htod(d_beta, b)

    threads = max(min(row_size, _MAX_THREADS), _WARP_SIZE)
    values_per_thread = (row_size + threads - 1) // threads
    kernel = _cached_kernels.get(values_per_thread, _generic_kernel)
    kernel(d_input, d_gamma, d_beta, d_output, np.int32(row_size), np.float32(eps),
           block=(threads, 1, 1), grid=(row_count, 1))

    output = np.empty_like(x)
    drv.memcpy_dtoh(output, d_output)

    d_input.free()
    d_output.free()
    d_gamma.free()
    d_beta.free()
    return output
