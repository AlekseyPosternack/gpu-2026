import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

_FAST_KERNEL = """
#define MAX_THREADS 256

__global__ void layernorm_fast_kernel(const float* __restrict__ input,
                                      const float* __restrict__ gamma,
                                      const float* __restrict__ beta,
                                      float* __restrict__ output,
                                      int row_size,
                                      float eps) {
    int row_idx = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    const float* row_in = input + row_idx * row_size;
    float* row_out = output + row_idx * row_size;

    float thread_sum = 0.0f;
    float thread_sum_sq = 0.0f;

    int row_size_v4 = row_size / 4;
    const float4* row_in_v4 = reinterpret_cast<const float4*>(row_in);

    for (int i = tid; i < row_size_v4; i += num_threads) {
        float4 val = row_in_v4[i];
        thread_sum += val.x + val.y + val.z + val.w;
        thread_sum_sq += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        thread_sum_sq += __shfl_down_sync(0xffffffff, thread_sum_sq, offset);
    }

    __shared__ float s_sum[8];
    __shared__ float s_sum_sq[8];

    int lane = tid & 31;
    int warpId = tid >> 5;

    if (lane == 0) {
        s_sum[warpId] = thread_sum;
        s_sum_sq[warpId] = thread_sum_sq;
    }
    __syncthreads();

    if (warpId == 0) {
        thread_sum = (tid < 8) ? s_sum[lane] : 0.0f;
        thread_sum_sq = (tid < 8) ? s_sum_sq[lane] : 0.0f;

        #pragma unroll
        for (int offset = 4; offset > 0; offset /= 2) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
            thread_sum_sq += __shfl_down_sync(0xffffffff, thread_sum_sq, offset);
        }
    }

    __shared__ float s_mean, s_inv_std;
    if (tid == 0) {
        float mean = thread_sum / row_size;
        float mean_sq = thread_sum_sq / row_size;
        float var = mean_sq - (mean * mean);
        if (var < 0.0f) var = 0.0f;
        s_mean = mean;
        s_inv_std = rsqrtf(var + eps);
    }
    __syncthreads();

    float mean = s_mean;
    float inv_std = s_inv_std;

    float4* row_out_v4 = reinterpret_cast<float4*>(row_out);
    const float4* gamma_v4 = reinterpret_cast<const float4*>(gamma);
    const float4* beta_v4 = reinterpret_cast<const float4*>(beta);

    for (int i = tid; i < row_size_v4; i += num_threads) {
        float4 in_val = row_in_v4[i];
        float4 g_val = gamma_v4[i];
        float4 b_val = beta_v4[i];
        float4 out_val;

        out_val.x = ((in_val.x - mean) * inv_std) * g_val.x + b_val.x;
        out_val.y = ((in_val.y - mean) * inv_std) * g_val.y + b_val.y;
        out_val.z = ((in_val.z - mean) * inv_std) * g_val.z + b_val.z;
        out_val.w = ((in_val.w - mean) * inv_std) * g_val.w + b_val.w;

        row_out_v4[i] = out_val;
    }
}
"""

_mod = SourceModule(_FAST_KERNEL)
_kernel_fn = _mod.get_function("layernorm_fast_kernel")

_d_input = None
_d_gamma = None
_d_beta = None
_d_output = None
_allocated_size = 0
_allocated_params = 0

def layernorm_pycuda(input_data, gamma, beta, row_size, eps=1e-5):
    global _d_input, _d_gamma, _d_beta, _d_output, _allocated_size, _allocated_params

    num_elements = input_data.size
    num_rows = num_elements // row_size
    output_arr = np.empty_like(input_data)

    input_bytes = input_data.nbytes
    param_bytes = gamma.nbytes

    if _d_input is None or _allocated_size < input_bytes:
        if _d_input is not None:
            _d_input.free()
            _d_output.free()
        _d_input = cuda.mem_alloc(input_bytes)
        _d_output = cuda.mem_alloc(input_bytes)
        _allocated_size = input_bytes

    if _d_gamma is None or _allocated_params < param_bytes:
        if _d_gamma is not None:
            _d_gamma.free()
            _d_beta.free()
        _d_gamma = cuda.mem_alloc(param_bytes)
        _d_beta = cuda.mem_alloc(param_bytes)
        _allocated_params = param_bytes

    cuda.memcpy_htod(_d_input, input_data)
    cuda.memcpy_htod(_d_gamma, gamma)
    cuda.memcpy_htod(_d_beta, beta)

    block_size = (256, 1, 1)
    grid_size = (num_rows, 1, 1)

    _kernel_fn(
        _d_input, _d_gamma, _d_beta, _d_output,
        np.int32(row_size), np.float32(eps),
        block=block_size, grid=grid_size
    )

    cuda.memcpy_dtoh(output_arr, _d_output)
    return output_arr