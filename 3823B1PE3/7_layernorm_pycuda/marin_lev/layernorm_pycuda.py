import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

cuda.init()
if cuda.Context.get_current() is None:
    import pycuda.autoinit


_source = r"""
#include <cuda_runtime.h>
#include <stddef.h>

__device__ __forceinline__ float warp_sum(float value) {
    #pragma unroll
    for (int step = 16; step > 0; step /= 2) {
        value += __shfl_down_sync(0xffffffffu, value, step);
    }
    return value;
}

__device__ __forceinline__ float block_sum(float value, float* partial) {
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    value = warp_sum(value);

    if (lane == 0) {
        partial[warp] = value;
    }
    __syncthreads();

    value = 0.0f;
    if (threadIdx.x < blockDim.x / 32) {
        value = partial[lane];
    }

    if (warp == 0) {
        value = warp_sum(value);
        if (lane == 0) {
            partial[0] = value;
        }
    }
    __syncthreads();

    float result = partial[0];
    __syncthreads();
    return result;
}

extern "C" __global__ void layer_norm(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int row_size,
    float eps) {

    extern __shared__ float values[];
    __shared__ float partial[8];

    size_t start = static_cast<size_t>(blockIdx.x) * row_size;
    bool use_shared = row_size <= 8192;

    float shift = input[start];
    float sum = 0.0f;

    for (int col = threadIdx.x; col < row_size; col += blockDim.x) {
        float x = input[start + col] - shift;

        if (use_shared) {
            values[col] = x;
        }

        sum += x;
    }

    float mean = block_sum(sum, partial) / row_size;

    float squares = 0.0f;

    for (int col = threadIdx.x; col < row_size; col += blockDim.x) {
        float x = use_shared ? values[col] : input[start + col] - shift;
        float difference = x - mean;

        if (use_shared) {
            values[col] = difference;
        } else {
            output[start + col] = difference;
        }

        squares += difference * difference;
    }

    float variance = block_sum(squares, partial) / row_size;
    float scale = rsqrtf(variance + eps);

    for (int col = threadIdx.x; col < row_size; col += blockDim.x) {
        float difference = use_shared ? values[col] : output[start + col];
        output[start + col] = difference * scale * gamma[col] + beta[col];
    }
}
"""

_module = None
_kernel = None
_context = None

_buffers = [None, None, None, None]
_sizes = [0, 0, 0, 0]


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    global _module, _kernel, _context, _buffers, _sizes

    x = np.ascontiguousarray(input, dtype=np.float32).reshape(-1)
    gamma = np.ascontiguousarray(gamma, dtype=np.float32).reshape(-1)
    beta = np.ascontiguousarray(beta, dtype=np.float32).reshape(-1)
    row_size = int(row_size)

    if x.size == 0:
        return x.copy()

    context = cuda.Context.get_current()

    if _kernel is None or context != _context:
        _module = SourceModule(_source, options=["-O3"], no_extern_c=True)
        _kernel = _module.get_function("layer_norm")

        _buffers = [None, None, None, None]
        _sizes = [0, 0, 0, 0]
        _context = context

    sizes = [x.nbytes, gamma.nbytes, beta.nbytes, x.nbytes]

    for i in range(4):
        if sizes[i] > _sizes[i]:
            _buffers[i] = cuda.mem_alloc(sizes[i])
            _sizes[i] = sizes[i]

    d_input, d_gamma, d_beta, d_output = _buffers

    cuda.memcpy_htod(d_input, x)
    cuda.memcpy_htod(d_gamma, gamma)
    cuda.memcpy_htod(d_beta, beta)

    threads = 32
    while threads < 256 and threads < row_size:
        threads *= 2

    rows = x.size // row_size
    shared_bytes = row_size * 4 if row_size <= 8192 else 0

    _kernel(
        d_input, d_gamma, d_beta, d_output,
        np.int32(row_size), np.float32(eps),
        block=(threads, 1, 1),
        grid=(rows, 1),
        shared=shared_bytes
    )

    output = np.empty_like(x)
    cuda.memcpy_dtoh(output, d_output)

    return output
