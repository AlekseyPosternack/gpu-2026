import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


cuda_code = """
__global__ void layernorm_kernel(const float* __restrict__ input,
                                 const float* __restrict__ gamma,
                                 const float* __restrict__ beta,
                                 float* __restrict__ output,
                                 int row_size,
                                 float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float sdata[];
    
    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (int i = tid; i < row_size; i += blockDim.x) {
        float val = input[row * row_size + i];
        sum += val;
        sum_sq += val * val;
    }
    
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / row_size;
    
    __syncthreads();
    sdata[tid] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float variance = (sdata[0] / row_size) - (mean * mean);
    float inv_std = rsqrtf(variance + eps);
    
    // 2. Применяем нормализацию
    for (int i = tid; i < row_size; i += blockDim.x) {
        float val = input[row * row_size + i];
        output[row * row_size + i] = (val - mean) * inv_std * gamma[i] + beta[i];
    }
}
"""

def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    """
    Apply Layer Normalization to each row of the input matrix.
    """
    input_np = np.array(input, dtype=np.float32)
    gamma_np = np.array(gamma, dtype=np.float32)
    beta_np = np.array(beta, dtype=np.float32)
    
    row_count = input_np.size // row_size
    output_np = np.empty_like(input_np)

    mod = SourceModule(cuda_code)
    layernorm_kernel = mod.get_function("layernorm_kernel")

    input_gpu = cuda.mem_alloc(input_np.nbytes)
    gamma_gpu = cuda.mem_alloc(gamma_np.nbytes)
    beta_gpu = cuda.mem_alloc(beta_np.nbytes)
    output_gpu = cuda.mem_alloc(output_np.nbytes)

    cuda.memcpy_htod(input_gpu, input_np)
    cuda.memcpy_htod(gamma_gpu, gamma_np)
    cuda.memcpy_htod(beta_gpu, beta_np)

    block_size = min(256, row_size)
    grid_size = (row_count, 1, 1)
    block = (block_size, 1, 1)
    shared_mem_size = block_size * np.dtype(np.float32).itemsize

    layernorm_kernel(input_gpu, gamma_gpu, beta_gpu, output_gpu,
                     np.int32(row_size), np.float32(eps),
                     block=block, grid=grid_size, shared=shared_mem_size)

    # Забираем результат
    cuda.memcpy_dtoh(output_np, output_gpu)
    
    return output_np