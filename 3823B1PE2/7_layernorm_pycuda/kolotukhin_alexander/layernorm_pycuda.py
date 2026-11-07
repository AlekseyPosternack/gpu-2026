import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

_KERNEL_CODE = r"""
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_reduce_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ __forceinline__ float block_reduce_sum(float value, float* shared_memory) {
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
    value = warp_reduce_sum(value);
    if (lane == 0) {
        shared_memory[warp_id] = value;
    }
    __syncthreads();
    
    value = (threadIdx.x < (blockDim.x / 32)) ? shared_memory[threadIdx.x] : 0.0f;
    if (warp_id == 0) {
        value = warp_reduce_sum(value);
        if (lane == 0) {
            shared_memory[0] = value;
        }
    }
    __syncthreads();
    return shared_memory[0];
}

__global__ void __launch_bounds__(256) layernorm_kernel(
    const float* __restrict__ input_matrix,
    const float* __restrict__ gamma_vector,
    const float* __restrict__ beta_vector,
    float* __restrict__ output_matrix,
    int row_size,
    float eps)
{
    __shared__ float warp_sums_for_sum[32];
    __shared__ float warp_sums_for_sq[32];
    __shared__ float shared_mean;
    __shared__ float shared_inv_std;

    const int row_index = blockIdx.x;
    const float* row_input = input_matrix + (size_t)row_index * row_size;
    float* row_output = output_matrix + (size_t)row_index * row_size;

    const float anchor_value = row_input[0];

    float sum = 0.0f;
    float sum_squares = 0.0f;

    const int vector_count = row_size / 4;
    const float4* input_vec4 = reinterpret_cast<const float4*>(row_input);

    for (int i = threadIdx.x; i < vector_count; i += blockDim.x) {
        float4 value_vec = input_vec4[i];
        value_vec.x -= anchor_value;
        value_vec.y -= anchor_value;
        value_vec.z -= anchor_value;
        value_vec.w -= anchor_value;
        
        sum += value_vec.x + value_vec.y + value_vec.z + value_vec.w;
        sum_squares += value_vec.x * value_vec.x + value_vec.y * value_vec.y + 
                       value_vec.z * value_vec.z + value_vec.w * value_vec.w;
    }

    for (int i = vector_count * 4 + threadIdx.x; i < row_size; i += blockDim.x) {
        float value = row_input[i] - anchor_value;
        sum += value;
        sum_squares += value * value;
    }

    sum = block_reduce_sum(sum, warp_sums_for_sum);
    sum_squares = block_reduce_sum(sum_squares, warp_sums_for_sq);

    if (threadIdx.x == 0) {
        float mean = sum / row_size;
        float variance = (sum_squares / row_size) - (mean * mean);
        shared_mean = mean;
        shared_inv_std = rsqrtf(variance + eps);
    }
    __syncthreads();

    float mean = shared_mean;
    float inv_std = shared_inv_std;

    float4* output_vec4 = reinterpret_cast<float4*>(row_output);
    const float4* gamma_vec4 = reinterpret_cast<const float4*>(gamma_vector);
    const float4* beta_vec4 = reinterpret_cast<const float4*>(beta_vector);

    for (int i = threadIdx.x; i < vector_count; i += blockDim.x) {
        float4 value_vec = input_vec4[i];
        float4 gamma_vec = gamma_vec4[i];
        float4 beta_vec = beta_vec4[i];

        float4 result_vec;
        result_vec.x = ((value_vec.x - anchor_value) - mean) * inv_std * gamma_vec.x + beta_vec.x;
        result_vec.y = ((value_vec.y - anchor_value) - mean) * inv_std * gamma_vec.y + beta_vec.y;
        result_vec.z = ((value_vec.z - anchor_value) - mean) * inv_std * gamma_vec.z + beta_vec.z;
        result_vec.w = ((value_vec.w - anchor_value) - mean) * inv_std * gamma_vec.w + beta_vec.w;
        
        output_vec4[i] = result_vec;
    }

    for (int i = vector_count * 4 + threadIdx.x; i < row_size; i += blockDim.x) {
        row_output[i] = ((row_input[i] - anchor_value) - mean) * inv_std * gamma_vector[i] + beta_vector[i];
    }
}
"""

_module = SourceModule(_KERNEL_CODE, options=["-O3", "--use_fast_math"])
_layernorm_kernel = _module.get_function("layernorm_kernel")

_buffers = {}

def _get_device_buffer(buffer_name, num_bytes):
    if buffer_name not in _buffers or _buffers[buffer_name][1] < num_bytes:
        _buffers[buffer_name] = (cuda.mem_alloc(num_bytes), num_bytes)
    return _buffers[buffer_name][0]

def _get_host_buffer(buffer_name, num_elements):
    if buffer_name not in _buffers or _buffers[buffer_name][1] < num_elements:
        _buffers[buffer_name] = (cuda.pagelocked_empty(num_elements, dtype=np.float32), num_elements)
    return _buffers[buffer_name][0]

def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    input_data = np.ascontiguousarray(input, dtype=np.float32).ravel()
    gamma_data = np.ascontiguousarray(gamma, dtype=np.float32).ravel()
    beta_data = np.ascontiguousarray(beta, dtype=np.float32).ravel()

    row_size = int(row_size)
    n = input_data.size
    if n == 0:
        return np.empty(0, dtype=np.float32)
    
    row_count = n // row_size

    device_input = _get_device_buffer("device_input", input_data.nbytes)
    device_gamma = _get_device_buffer("device_gamma", gamma_data.nbytes)
    device_beta = _get_device_buffer("device_beta", beta_data.nbytes)
    device_output = _get_device_buffer("device_output", input_data.nbytes)
    
    host_output = _get_host_buffer("host_output", n)[:n]

    cuda.memcpy_htod(device_input, input_data)
    cuda.memcpy_htod(device_gamma, gamma_data)
    cuda.memcpy_htod(device_beta, beta_data)

    _layernorm_kernel(
        device_input, device_gamma, device_beta, device_output,
        np.int32(row_size), np.float32(eps),
        block=(256, 1, 1), grid=(row_count, 1, 1)
    )

    cuda.memcpy_dtoh(host_output, device_output)
    
    return host_output