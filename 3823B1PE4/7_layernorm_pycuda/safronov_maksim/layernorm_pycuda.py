import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


_source = r"""
extern "C" __global__ void layernorm_kernel(const float* input, const float* gamma, const float* beta, float* output, int row_size, float eps) {
    __shared__ float partial[32];
    __shared__ float row_mean;
    __shared__ float row_inverse_std;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const unsigned long long offset = (unsigned long long)blockIdx.x * row_size;

    float sum = 0.0f;
    for (int column = threadIdx.x; column < row_size; column += blockDim.x) {
        sum += input[offset + column];
    }

#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, delta);
    }

    if (lane == 0) {
        partial[warp] = sum;
    }
    __syncthreads();

    if (warp == 0) {
        sum = lane < 8 ? partial[lane] : 0.0f;
#pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, delta);
        }
        if (lane == 0) {
            row_mean = sum / row_size;
        }
    }
    __syncthreads();

    float square_sum = 0.0f;
    for (int column = threadIdx.x; column < row_size; column += blockDim.x) {
        const float difference = input[offset + column] - row_mean;
        square_sum = fmaf(difference, difference, square_sum);
    }

#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        square_sum += __shfl_down_sync(0xffffffff, square_sum, delta);
    }

    if (lane == 0) {
        partial[warp] = square_sum;
    }
    __syncthreads();

    if (warp == 0) {
        square_sum = lane < 8 ? partial[lane] : 0.0f;
#pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1) {
            square_sum += __shfl_down_sync(0xffffffff, square_sum, delta);
        }
        if (lane == 0) {
            row_inverse_std = rsqrtf(square_sum / row_size + eps);
        }
    }
    __syncthreads();

    for (int column = threadIdx.x; column < row_size; column += blockDim.x) {
        output[offset + column] = (input[offset + column] - row_mean) * row_inverse_std * gamma[column] + beta[column];
    }
}
"""


_module = SourceModule(_source, options=["-O3"])
_kernel = _module.get_function("layernorm_kernel")
_buffers = {}


def _reserve(name, size):
    allocation = _buffers.get(name)
    if allocation is None or allocation[1] < size:
        allocation = (cuda.mem_alloc(size), size)
        _buffers[name] = allocation
    return allocation[0]


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    if row_size <= 0:
        raise ValueError("row_size must be positive")

    input_array = np.ascontiguousarray(input, dtype=np.float32).reshape(-1)
    gamma_array = np.ascontiguousarray(gamma, dtype=np.float32).reshape(-1)
    beta_array = np.ascontiguousarray(beta, dtype=np.float32).reshape(-1)

    if input_array.size % row_size != 0:
        raise ValueError("input size must be divisible by row_size")
    if gamma_array.size != row_size or beta_array.size != row_size:
        raise ValueError("gamma and beta sizes must match row_size")
    if input_array.size == 0:
        return np.empty(0, dtype=np.float32)

    output = np.empty_like(input_array)
    input_device = _reserve("input", input_array.nbytes)
    gamma_device = _reserve("gamma", gamma_array.nbytes)
    beta_device = _reserve("beta", beta_array.nbytes)

    cuda.memcpy_htod(input_device, input_array)
    cuda.memcpy_htod(gamma_device, gamma_array)
    cuda.memcpy_htod(beta_device, beta_array)

    row_count = input_array.size // row_size
    _kernel(input_device, gamma_device, beta_device, input_device, np.int32(row_size), np.float32(eps), block=(256, 1, 1), grid=(int(row_count), 1, 1))
    cuda.memcpy_dtoh(output, input_device)
    return output
