import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


_KERNEL_SOURCE = r"""
extern "C" __global__ void layernorm_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int row_size,
    float eps
) {
    __shared__ float reduction[256];

    const int row = blockIdx.x;
    const int thread = threadIdx.x;
    const int row_start = row * row_size;

    float local_sum = 0.0f;
    for (int col = thread; col < row_size; col += blockDim.x) {
        local_sum += input[row_start + col];
    }

    reduction[thread] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread < stride) {
            reduction[thread] += reduction[thread + stride];
        }
        __syncthreads();
    }

    const float mean = reduction[0] / row_size;

    float local_variance = 0.0f;
    for (int col = thread; col < row_size; col += blockDim.x) {
        const float diff = input[row_start + col] - mean;
        local_variance += diff * diff;
    }

    reduction[thread] = local_variance;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread < stride) {
            reduction[thread] += reduction[thread + stride];
        }
        __syncthreads();
    }

    const float variance = reduction[0] / row_size;
    const float inv_std = rsqrtf(variance + eps);

    for (int col = thread; col < row_size; col += blockDim.x) {
        const float normalized = (input[row_start + col] - mean) * inv_std;
        output[row_start + col] = normalized * gamma[col] + beta[col];
    }
}
"""

_kernel = None


def _get_kernel():
    global _kernel

    if _kernel is None:
        module = SourceModule(_KERNEL_SOURCE)
        _kernel = module.get_function("layernorm_kernel")

    return _kernel


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    if row_size <= 0:
        raise ValueError("row_size must be positive")

    if row_size & (row_size - 1):
        raise ValueError("row_size must be a power of 2")

    input_array = np.ascontiguousarray(input, dtype=np.float32).reshape(-1)
    gamma_array = np.ascontiguousarray(gamma, dtype=np.float32).reshape(-1)
    beta_array = np.ascontiguousarray(beta, dtype=np.float32).reshape(-1)

    if input_array.size % row_size != 0:
        raise ValueError("Input size must be divisible by row_size")

    if gamma_array.size != row_size or beta_array.size != row_size:
        raise ValueError("gamma and beta must each have row_size elements")

    if input_array.size == 0:
        return np.empty(0, dtype=np.float32)

    row_count = input_array.size // row_size
    output = np.empty_like(input_array)

    device_input = cuda.mem_alloc(input_array.nbytes)
    device_gamma = cuda.mem_alloc(gamma_array.nbytes)
    device_beta = cuda.mem_alloc(beta_array.nbytes)
    device_output = cuda.mem_alloc(output.nbytes)

    cuda.memcpy_htod(device_input, input_array)
    cuda.memcpy_htod(device_gamma, gamma_array)
    cuda.memcpy_htod(device_beta, beta_array)

    kernel = _get_kernel()
    kernel(
        device_input,
        device_gamma,
        device_beta,
        device_output,
        np.int32(row_size),
        np.float32(eps),
        block=(256, 1, 1),
        grid=(row_count, 1, 1),
    )

    cuda.memcpy_dtoh(output, device_output)
    return output
