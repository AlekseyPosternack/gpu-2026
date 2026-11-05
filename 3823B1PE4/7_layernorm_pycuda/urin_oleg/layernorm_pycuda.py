"""Row-wise float32 layer normalization implemented with PyCUDA."""

from functools import lru_cache
from operator import index

import numpy as np


_CUDA_SOURCE = r"""
#include <math.h>
#include <stddef.h>

extern "C" __global__ void layernorm_rows(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int row_size,
    float eps
) {
    __shared__ float scratch[256];
    const unsigned int tid = threadIdx.x;
    const unsigned int stride = blockDim.x;
    const size_t offset = (size_t)blockIdx.x * (size_t)row_size;
    const bool one_value_per_thread = row_size <= (int)stride;

    // For short rows, keep the input value in a register across both reductions.
    float value = 0.0f;
    float local_sum = 0.0f;
    if (one_value_per_thread) {
        if (tid < (unsigned int)row_size) {
            value = input[offset + tid];
            local_sum = value;
        }
    } else {
        for (size_t col = tid; col < (size_t)row_size; col += stride) {
            local_sum += input[offset + col];
        }
    }

    scratch[tid] = local_sum;
    __syncthreads();
    for (unsigned int step = stride / 2; step > 0; step >>= 1) {
        if (tid < step) {
            scratch[tid] += scratch[tid + step];
        }
        __syncthreads();
    }
    const float mean = scratch[0] / (float)row_size;
    __syncthreads(); // Finish reading the mean before reusing scratch.

    float local_squared_diff = 0.0f;
    if (one_value_per_thread) {
        if (tid < (unsigned int)row_size) {
            const float diff = value - mean;
            local_squared_diff = diff * diff;
        }
    } else {
        for (size_t col = tid; col < (size_t)row_size; col += stride) {
            const float diff = input[offset + col] - mean;
            local_squared_diff += diff * diff;
        }
    }

    scratch[tid] = local_squared_diff;
    __syncthreads();
    for (unsigned int step = stride / 2; step > 0; step >>= 1) {
        if (tid < step) {
            scratch[tid] += scratch[tid + step];
        }
        __syncthreads();
    }
    const float inverse_std = rsqrtf(scratch[0] / (float)row_size + eps);

    if (one_value_per_thread) {
        if (tid < (unsigned int)row_size) {
            output[offset + tid] =
                gamma[tid] * ((value - mean) * inverse_std) + beta[tid];
        }
    } else {
        for (size_t col = tid; col < (size_t)row_size; col += stride) {
            output[offset + col] = gamma[col] *
                ((input[offset + col] - mean) * inverse_std) + beta[col];
        }
    }
}
"""


@lru_cache(maxsize=1)
def _get_kernel():
    import pycuda.driver as cuda

    cuda.init()
    if cuda.Context.get_current() is None:
        import pycuda.autoinit

    from pycuda.compiler import SourceModule

    module = SourceModule(_CUDA_SOURCE, options=["-O3"], no_extern_c=True)
    return module, module.get_function("layernorm_rows")


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    """Apply float32 LayerNorm independently to each row of a flattened matrix.

    The returned NumPy array is flat and has the same number of elements as
    ``input``. ``gamma`` and ``beta`` each contain one value per feature.
    """
    row_size = index(row_size)
    if row_size <= 0:
        raise ValueError("row_size must be positive")
    if row_size > np.iinfo(np.int32).max:
        raise ValueError("row_size exceeds the CUDA kernel's int32 limit")

    eps = np.float32(eps)
    if not np.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and nonnegative")

    values = np.ascontiguousarray(np.asarray(input, dtype=np.float32).reshape(-1))
    scales = np.ascontiguousarray(np.asarray(gamma, dtype=np.float32).reshape(-1))
    shifts = np.ascontiguousarray(np.asarray(beta, dtype=np.float32).reshape(-1))
    if values.size % row_size:
        raise ValueError("input length must be divisible by row_size")
    if scales.size != row_size or shifts.size != row_size:
        raise ValueError("gamma and beta must each have row_size elements")
    if values.size == 0:
        return np.empty(0, dtype=np.float32)

    row_count = values.size // row_size
    if row_count > np.iinfo(np.int32).max:
        raise ValueError("row count exceeds the CUDA grid limit")

    _, kernel = _get_kernel()
    import pycuda.gpuarray as gpuarray

    device_input = gpuarray.to_gpu(values)
    device_gamma = gpuarray.to_gpu(scales)
    device_beta = gpuarray.to_gpu(shifts)
    device_output = gpuarray.empty(values.shape, np.float32)

    threads = 1
    while threads < row_size and threads < 256:
        threads <<= 1
    kernel(
        device_input.gpudata,
        device_gamma.gpudata,
        device_beta.gpudata,
        device_output.gpudata,
        np.int32(row_size),
        eps,
        block=(threads, 1, 1),
        grid=(int(row_count), 1, 1),
    )
    return device_output.get()
