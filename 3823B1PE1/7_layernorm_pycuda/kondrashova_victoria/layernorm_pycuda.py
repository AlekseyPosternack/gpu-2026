import numpy as np

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


KERNEL_CODE = """
__global__ void layernorm_kernel(const float *input,
                                 const float *gamma,
                                 const float *beta,
                                 float *output,
                                 int row_size,
                                 float eps) {
    __shared__ float shared[1024];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float *row_input = input + row * row_size;
    float *row_output = output + row * row_size;

    float local_sum = 0.0f;

    for (int i = tid; i < row_size; i += blockDim.x) {
        local_sum += row_input[i];
    }

    shared[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    float mean = shared[0] / row_size;
    __syncthreads();

    float local_var = 0.0f;

    for (int i = tid; i < row_size; i += blockDim.x) {
        float diff = row_input[i] - mean;
        local_var += diff * diff;
    }

    shared[tid] = local_var;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    float variance = shared[0] / row_size;
    float std_dev = sqrtf(variance + eps);
    __syncthreads();

    for (int i = tid; i < row_size; i += blockDim.x) {
        float normalized = (row_input[i] - mean) / std_dev;
        row_output[i] = gamma[i] * normalized + beta[i];
    }
}
"""


_mod = SourceModule(KERNEL_CODE)
_layernorm_kernel = _mod.get_function("layernorm_kernel")


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    """
    Apply Layer Normalization to each row of the input matrix.
    """
    input = np.asarray(input, dtype=np.float32)
    gamma = np.asarray(gamma, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)

    input = input.flatten()
    gamma = gamma.flatten()
    beta = beta.flatten()

    total_size = input.size
    row_count = total_size // row_size

    output = np.empty_like(input)

    d_input = cuda.mem_alloc(input.nbytes)
    d_gamma = cuda.mem_alloc(gamma.nbytes)
    d_beta = cuda.mem_alloc(beta.nbytes)
    d_output = cuda.mem_alloc(output.nbytes)

    cuda.memcpy_htod(d_input, input)
    cuda.memcpy_htod(d_gamma, gamma)
    cuda.memcpy_htod(d_beta, beta)

    block_size = 256

    _layernorm_kernel(
        d_input,
        d_gamma,
        d_beta,
        d_output,
        np.int32(row_size),
        np.float32(eps),
        block=(block_size, 1, 1),
        grid=(row_count, 1, 1),
    )

    cuda.memcpy_dtoh(output, d_output)

    return output
