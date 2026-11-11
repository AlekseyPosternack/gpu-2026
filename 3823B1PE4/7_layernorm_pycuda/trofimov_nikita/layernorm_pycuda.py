
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


kernel_code = r"""
__global__ void layernorm_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int row_size,
    float eps)
{
    int tid = threadIdx.x;
    int row = blockIdx.x;

    const float* row_input = input + (size_t)row * row_size;
    float* row_output = output + (size_t)row * row_size;

    __shared__ float shared[256];

    // Вычисляем сумму элементов строки.
    float sum = 0.0f;

    for (int i = tid; i < row_size; i += blockDim.x) {
        sum += row_input[i];
    }

    shared[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    float mean = shared[0] / row_size;

    float variance_sum = 0.0f;

    for (int i = tid; i < row_size; i += blockDim.x) {
        float diff = row_input[i] - mean;
        variance_sum += diff * diff;
    }

    shared[tid] = variance_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    float variance = shared[0] / row_size;
    float inv_std = rsqrtf(variance + eps);

    for (int i = tid; i < row_size; i += blockDim.x) {
        float normalized = (row_input[i] - mean) * inv_std;
        row_output[i] = normalized * gamma[i] + beta[i];
    }
}
"""

module = SourceModule(kernel_code, options=["-O3"])
layernorm_kernel = module.get_function("layernorm_kernel")


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):

    input = np.ascontiguousarray(input, dtype=np.float32).reshape(-1)
    gamma = np.ascontiguousarray(gamma, dtype=np.float32).reshape(-1)
    beta = np.ascontiguousarray(beta, dtype=np.float32).reshape(-1)

    row_size = int(row_size)

    if row_size <= 0 or (row_size & (row_size - 1)) != 0:
        raise ValueError("row_size must be a power of 2")

    if input.size % row_size != 0:
        raise ValueError("input size must be divisible by row_size")

    if gamma.size != row_size or beta.size != row_size:
        raise ValueError("gamma and beta must have length row_size")

    if input.size == 0:
        return np.empty(0, dtype=np.float32)

    num_rows = input.size // row_size

    output = np.empty_like(input)

    d_input = cuda.mem_alloc(input.nbytes)
    d_gamma = cuda.mem_alloc(gamma.nbytes)
    d_beta = cuda.mem_alloc(beta.nbytes)
    d_output = cuda.mem_alloc(output.nbytes)

    cuda.memcpy_htod(d_input, input)
    cuda.memcpy_htod(d_gamma, gamma)
    cuda.memcpy_htod(d_beta, beta)

    block_size = min(row_size, 256)

    layernorm_kernel(
        d_input,
        d_gamma,
        d_beta,
        d_output,
        np.int32(row_size),
        np.float32(eps),
        block=(block_size, 1, 1),
        grid=(num_rows, 1, 1)
    )

    cuda.memcpy_dtoh(output, d_output)

    d_input.free()
    d_gamma.free()
    d_beta.free()
    d_output.free()

    return output
