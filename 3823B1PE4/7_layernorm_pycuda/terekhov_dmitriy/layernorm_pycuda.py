import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda.tools import context_dependent_memoize


@context_dependent_memoize
def _get_kernel():
    module = SourceModule(r"""
    __global__ void layernorm(float* data, const float* gamma,
                              const float* beta, int row_size, float eps) {
        __shared__ float partial[256];
        int tid = threadIdx.x;
        size_t start = (size_t)blockIdx.x * row_size;
        float anchor = data[start];

        float sum = 0.0f;
        for (int col = tid; col < row_size; col += blockDim.x) {
            sum += data[start + col] - anchor;
        }
        partial[tid] = sum;
        __syncthreads();
        for (int step = 128; step > 0; step /= 2) {
            if (tid < step) partial[tid] += partial[tid + step];
            __syncthreads();
        }
        float mean_offset = partial[0] / row_size;
        __syncthreads();

        float squared_sum = 0.0f;
        for (int col = tid; col < row_size; col += blockDim.x) {
            float centered = (data[start + col] - anchor) - mean_offset;
            squared_sum += centered * centered;
        }
        partial[tid] = squared_sum;
        __syncthreads();
        for (int step = 128; step > 0; step /= 2) {
            if (tid < step) partial[tid] += partial[tid + step];
            __syncthreads();
        }
        float inverse_std = rsqrtf(partial[0] / row_size + eps);
        __syncthreads();

        for (int col = tid; col < row_size; col += blockDim.x) {
            float centered = (data[start + col] - anchor) - mean_offset;
            data[start + col] = centered * inverse_std * gamma[col] + beta[col];
        }
    }
    """, options=["-O3"])
    return module.get_function("layernorm")


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    if not isinstance(row_size, (int, np.integer)) or row_size <= 0:
        raise ValueError("row_size must be a positive integer")
    if row_size > np.iinfo(np.int32).max:
        raise ValueError("row_size is too large")
    eps = np.float32(eps)
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")

    values = np.ascontiguousarray(np.asarray(input, dtype=np.float32).reshape(-1))
    scale = np.ascontiguousarray(np.asarray(gamma, dtype=np.float32).reshape(-1))
    shift = np.ascontiguousarray(np.asarray(beta, dtype=np.float32).reshape(-1))
    if values.size % row_size != 0 or scale.size != row_size or shift.size != row_size:
        raise ValueError("Input dimensions do not match row_size")
    if values.size == 0:
        return np.empty(0, dtype=np.float32)

    kernel = _get_kernel()
    buffers = []
    try:
        for array in (values, scale, shift):
            buffers.append(cuda.to_device(array))
        kernel(buffers[0], buffers[1], buffers[2], np.int32(row_size), eps,
               block=(256, 1, 1), grid=(values.size // row_size, 1))
        result = np.empty_like(values)
        cuda.memcpy_dtoh(result, buffers[0])
        return result
    finally:
        for buffer in buffers:
            buffer.free()
