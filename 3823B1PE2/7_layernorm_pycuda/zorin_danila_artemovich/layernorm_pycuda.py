import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

_BLOCK_SIZE = 256
_KERNEL = SourceModule(r"""
extern "C" __global__ void layernorm_kernel(const float* input, const float* gamma,
    const float* beta, float* output, int row_size, float eps) {
    const int row = blockIdx.x, tid = threadIdx.x;
    extern __shared__ float shared[];
    float* sums = shared; float* squares = shared + blockDim.x;
    float sum = 0.0f;
    for (int col = tid; col < row_size; col += blockDim.x) sum += input[row * row_size + col];
    sums[tid] = sum; __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sums[tid] += sums[tid + stride]; __syncthreads();
    }
    const float mean = sums[0] / row_size;
    float square_sum = 0.0f;
    for (int col = tid; col < row_size; col += blockDim.x) {
        const float d = input[row * row_size + col] - mean; square_sum += d * d;
    }
    squares[tid] = square_sum; __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) squares[tid] += squares[tid + stride]; __syncthreads();
    }
    const float inv_std = rsqrtf(squares[0] / row_size + eps);
    for (int col = tid; col < row_size; col += blockDim.x) {
        const float normalized = (input[row * row_size + col] - mean) * inv_std;
        output[row * row_size + col] = normalized * gamma[col] + beta[col];
    }
}
""").get_function("layernorm_kernel")

def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    if row_size <= 0: raise ValueError("row_size must be positive")
    values = np.asarray(input, dtype=np.float32).reshape(-1)
    scale = np.asarray(gamma, dtype=np.float32).reshape(-1)
    shift = np.asarray(beta, dtype=np.float32).reshape(-1)
    if values.size % row_size != 0: raise ValueError("input size must be divisible by row_size")
    if scale.size != row_size or shift.size != row_size: raise ValueError("gamma and beta must have row_size elements")
    output = np.empty_like(values)
    if values.size == 0: return output
    device_input = cuda.mem_alloc(values.nbytes); device_gamma = cuda.mem_alloc(scale.nbytes)
    device_beta = cuda.mem_alloc(shift.nbytes); device_output = cuda.mem_alloc(output.nbytes)
    try:
        cuda.memcpy_htod(device_input, values); cuda.memcpy_htod(device_gamma, scale); cuda.memcpy_htod(device_beta, shift)
        _KERNEL(device_input, device_gamma, device_beta, device_output, np.int32(row_size), np.float32(eps),
                block=(_BLOCK_SIZE, 1, 1), grid=(values.size // row_size, 1, 1),
                shared=2 * _BLOCK_SIZE * np.dtype(np.float32).itemsize)
        cuda.Context.synchronize(); cuda.memcpy_dtoh(output, device_output)
    finally:
        device_output.free(); device_beta.free(); device_gamma.free(); device_input.free()
    return output
