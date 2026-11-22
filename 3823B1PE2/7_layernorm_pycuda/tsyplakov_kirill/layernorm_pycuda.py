import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from pycuda import gpuarray

_KERNEL_CODE = r"""
#define BLOCK_SIZE 256

__global__ void layernorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int row_size,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* row_in = input + (size_t)row * row_size;
    float* row_out = output + (size_t)row * row_size;

    extern __shared__ float smem[];
    float* reduce_buf = smem;
    float* row_cache  = smem + BLOCK_SIZE;

    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;
    for (int i = tid; i < row_size; i += blockDim.x) {
        float v = row_in[i];
        row_cache[i] = v;
        local_sum += v;
        local_sq_sum += v * v;
    }

    reduce_buf[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
        __syncthreads();
    }
    float sum = reduce_buf[0];

    __syncthreads();
    reduce_buf[tid] = local_sq_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
        __syncthreads();
    }
    float sq_sum = reduce_buf[0];

    float mean = sum / row_size;
    float var = sq_sum / row_size - mean * mean;
    float inv_std = rsqrtf(var + eps);

    for (int i = tid; i < row_size; i += blockDim.x) {
        float norm = (row_cache[i] - mean) * inv_std;
        row_out[i] = norm * gamma[i] + beta[i];
    }
}
"""

_mod = SourceModule(_KERNEL_CODE, options=['-ccbin=/usr/bin/g++-13'])
_layernorm_kernel = _mod.get_function("layernorm_kernel")


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    input_np = np.asarray(input, dtype=np.float32)
    gamma_np = np.asarray(gamma, dtype=np.float32)
    beta_np = np.asarray(beta, dtype=np.float32)

    total = input_np.size
    row_count = total // row_size

    input_gpu = gpuarray.to_gpu(input_np)
    gamma_gpu = gpuarray.to_gpu(gamma_np)
    beta_gpu = gpuarray.to_gpu(beta_np)
    output_gpu = gpuarray.empty_like(input_gpu)

    block_size = 256
    smem_bytes = (block_size + row_size) * 4

    _layernorm_kernel(
        input_gpu, gamma_gpu, beta_gpu, output_gpu,
        np.int32(row_size),
        np.float32(eps),
        block=(block_size, 1, 1),
        grid=(row_count, 1, 1),
        shared=smem_bytes
    )

    return output_gpu.get()
