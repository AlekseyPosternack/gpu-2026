"""CUDA LayerNorm implementation using PyCUDA."""

import numpy as np


_BLOCK_SIZE = 256
_MODULE = None
_KERNEL = None


_KERNEL_SOURCE = r"""
extern "C" __global__ void layernorm_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int row_size,
    float eps)
{
    __shared__ float reduction[256];
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int row_offset = row * row_size;

    float local_sum = 0.0f;
    for (int col = tid; col < row_size; col += blockDim.x) {
        local_sum += input[row_offset + col];
    }

    reduction[tid] = local_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            reduction[tid] += reduction[tid + offset];
        }
        __syncthreads();
    }
    const float mean = reduction[0] / (float)row_size;

    float local_variance = 0.0f;
    for (int col = tid; col < row_size; col += blockDim.x) {
        const float diff = input[row_offset + col] - mean;
        local_variance = fmaf(diff, diff, local_variance);
    }

    reduction[tid] = local_variance;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            reduction[tid] += reduction[tid + offset];
        }
        __syncthreads();
    }
    const float inv_std = rsqrtf(reduction[0] / (float)row_size + eps);

    for (int col = tid; col < row_size; col += blockDim.x) {
        const float normalized =
            (input[row_offset + col] - mean) * inv_std;
        output[row_offset + col] = normalized * gamma[col] + beta[col];
    }
}
"""


def _get_kernel():
    global _MODULE, _KERNEL
    if _KERNEL is None:
        import pycuda.autoinit  # noqa: F401  (creates the CUDA context)
        from pycuda.compiler import SourceModule

        _MODULE = SourceModule(_KERNEL_SOURCE, options=["-O3"])
        _KERNEL = _MODULE.get_function("layernorm_kernel")
    return _KERNEL


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    """Apply Layer Normalization independently to every row on the GPU."""
    if not isinstance(row_size, (int, np.integer)) or row_size <= 0:
        raise ValueError("row_size must be a positive integer")
    if eps < 0:
        raise ValueError("eps must be non-negative")

    input_array = np.ascontiguousarray(input, dtype=np.float32).reshape(-1)
    gamma_array = np.ascontiguousarray(gamma, dtype=np.float32).reshape(-1)
    beta_array = np.ascontiguousarray(beta, dtype=np.float32).reshape(-1)

    if input_array.size % row_size != 0:
        raise ValueError("input length must be divisible by row_size")
    if gamma_array.size != row_size or beta_array.size != row_size:
        raise ValueError("gamma and beta must have length row_size")

    result = np.empty_like(input_array)
    if input_array.size == 0:
        return result

    import pycuda.driver as cuda

    # Initialize the context and compile the kernel before allocating device
    # buffers. This also lets repeated calls reuse the compiled module.
    kernel = _get_kernel()
    row_count = input_array.size // row_size
    device_input = cuda.mem_alloc(input_array.nbytes)
    device_gamma = cuda.mem_alloc(gamma_array.nbytes)
    device_beta = cuda.mem_alloc(beta_array.nbytes)
    device_output = cuda.mem_alloc(result.nbytes)

    try:
        cuda.memcpy_htod(device_input, input_array)
        cuda.memcpy_htod(device_gamma, gamma_array)
        cuda.memcpy_htod(device_beta, beta_array)

        kernel(device_input, device_gamma, device_beta, device_output,
               np.int32(row_size), np.float32(eps),
               block=(_BLOCK_SIZE, 1, 1), grid=(int(row_count), 1, 1))
        cuda.memcpy_dtoh(result, device_output)
    finally:
        device_output.free()
        device_beta.free()
        device_gamma.free()
        device_input.free()

    return result
