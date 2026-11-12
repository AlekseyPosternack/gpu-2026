import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

_CUDA_KERNEL_SRC = r"""
#define BLOCK_THREADS 256
#define WARP_WIDTH 32
#define NUM_WARPS (BLOCK_THREADS / WARP_WIDTH)

__device__ __forceinline__ float warp_reduce_add(float val) {
    #pragma unroll
    for (int mask = WARP_WIDTH / 2; mask > 0; mask >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, mask);
    }
    return val;
}

extern "C" __global__ void layernorm_inplace_vec4(
    float* __restrict__ data,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int row_size,
    float eps
) {
    __shared__ float warp_scratch[NUM_WARPS];
    __shared__ float s_mean;
    __shared__ float s_inv_std;

    const int tid = threadIdx.x;
    const int lane = tid & (WARP_WIDTH - 1);
    const int wid = tid >> 5;

    float* const row_ptr = data + static_cast<size_t>(blockIdx.x) * row_size;
    const int vec_cols = row_size >> 2;
    const int tail_start = vec_cols << 2;
    const float inv_n = 1.0f / static_cast<float>(row_size);

    float4* const row_vec = reinterpret_cast<float4*>(row_ptr);
    const float4* const gamma_vec = reinterpret_cast<const float4*>(gamma);
    const float4* const beta_vec = reinterpret_cast<const float4*>(beta);

    float local_sum = 0.0f;
    for (int i = tid; i < vec_cols; i += BLOCK_THREADS) {
        const float4 v = row_vec[i];
        local_sum += (v.x + v.y) + (v.z + v.w);
    }
    for (int i = tail_start + tid; i < row_size; i += BLOCK_THREADS) {
        local_sum += row_ptr[i];
    }

    local_sum = warp_reduce_add(local_sum);
    if (lane == 0) {
        warp_scratch[wid] = local_sum;
    }
    __syncthreads();

    if (wid == 0) {
        float block_sum = (lane < NUM_WARPS) ? warp_scratch[lane] : 0.0f;
        block_sum = warp_reduce_add(block_sum);
        if (lane == 0) {
            s_mean = block_sum * inv_n;
        }
    }
    __syncthreads();

    const float mean = s_mean;

    float local_sq_diff = 0.0f;
    for (int i = tid; i < vec_cols; i += BLOCK_THREADS) {
        const float4 v = row_vec[i];
        const float dx = v.x - mean;
        const float dy = v.y - mean;
        const float dz = v.z - mean;
        const float dw = v.w - mean;
        local_sq_diff = fmaf(dx, dx, local_sq_diff);
        local_sq_diff = fmaf(dy, dy, local_sq_diff);
        local_sq_diff = fmaf(dz, dz, local_sq_diff);
        local_sq_diff = fmaf(dw, dw, local_sq_diff);
    }
    for (int i = tail_start + tid; i < row_size; i += BLOCK_THREADS) {
        const float d = row_ptr[i] - mean;
        local_sq_diff = fmaf(d, d, local_sq_diff);
    }

    local_sq_diff = warp_reduce_add(local_sq_diff);
    if (lane == 0) {
        warp_scratch[wid] = local_sq_diff;
    }
    __syncthreads();

    if (wid == 0) {
        float block_var = (lane < NUM_WARPS) ? warp_scratch[lane] : 0.0f;
        block_var = warp_reduce_add(block_var);
        if (lane == 0) {
            s_inv_std = rsqrtf(block_var * inv_n + eps);
        }
    }
    __syncthreads();

    const float inv_std = s_inv_std;

    for (int i = tid; i < vec_cols; i += BLOCK_THREADS) {
        const float4 v = row_vec[i];
        const float4 g = gamma_vec[i];
        const float4 b = beta_vec[i];

        float4 res;
        res.x = fmaf((v.x - mean) * inv_std, g.x, b.x);
        res.y = fmaf((v.y - mean) * inv_std, g.y, b.y);
        res.z = fmaf((v.z - mean) * inv_std, g.z, b.z);
        res.w = fmaf((v.w - mean) * inv_std, g.w, b.w);
        row_vec[i] = res;
    }
    for (int i = tail_start + tid; i < row_size; i += BLOCK_THREADS) {
        row_ptr[i] = fmaf((row_ptr[i] - mean) * inv_std, gamma[i], beta[i]);
    }
}
"""

_compiled_mod = SourceModule(_CUDA_KERNEL_SRC, options=["-O3"], no_extern_c=True)
_ln_kernel = _compiled_mod.get_function("layernorm_inplace_vec4")
_ln_kernel.prepare("PPPif")

_gpu_data_buf = None
_gpu_data_bytes = 0
_gpu_gamma_buf = None
_gpu_beta_buf = None
_gpu_param_bytes = 0


def _acquire_device_memory(data_nbytes, param_nbytes):
    global _gpu_data_buf, _gpu_data_bytes
    global _gpu_gamma_buf, _gpu_beta_buf, _gpu_param_bytes

    if data_nbytes > _gpu_data_bytes:
        if _gpu_data_buf is not None:
            _gpu_data_buf.free()
        _gpu_data_buf = cuda.mem_alloc(data_nbytes)
        _gpu_data_bytes = data_nbytes

    if param_nbytes > _gpu_param_bytes:
        if _gpu_gamma_buf is not None:
            _gpu_gamma_buf.free()
        if _gpu_beta_buf is not None:
            _gpu_beta_buf.free()
        _gpu_gamma_buf = cuda.mem_alloc(param_nbytes)
        _gpu_beta_buf = cuda.mem_alloc(param_nbytes)
        _gpu_param_bytes = param_nbytes


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    """
    Apply Layer Normalization to each row of the input matrix.

    Parameters
    ----------
    input : list or numpy.ndarray of float
        Flattened matrix in row-major order. Its length must be divisible by row_size.
    gamma : list or numpy.ndarray of float
        Scale parameter, length = row_size.
    beta : list or numpy.ndarray of float
        Shift parameter, length = row_size.
    row_size : int
        Number of features per row (i.e., number of columns).
    eps : float, optional
        Small constant for numerical stability.

    Returns
    -------
    numpy.ndarray
        Flattened matrix of the same shape as input, containing the row-wise
        normalized results.
    """
    row_size = int(row_size)
    x_host = np.ascontiguousarray(input, dtype=np.float32).ravel()
    if x_host.size == 0 or row_size <= 0:
        return np.empty(0, dtype=np.float32)

    g_host = np.ascontiguousarray(gamma, dtype=np.float32).ravel()
    b_host = np.ascontiguousarray(beta, dtype=np.float32).ravel()

    row_count = x_host.size // row_size
    _acquire_device_memory(x_host.nbytes, g_host.nbytes)

    cuda.memcpy_htod(_gpu_data_buf, x_host)
    cuda.memcpy_htod(_gpu_gamma_buf, g_host)
    cuda.memcpy_htod(_gpu_beta_buf, b_host)

    _ln_kernel.prepared_call(
        (row_count, 1, 1),
        (256, 1, 1),
        _gpu_data_buf,
        _gpu_gamma_buf,
        _gpu_beta_buf,
        np.int32(row_size),
        np.float32(eps),
    )

    result = np.empty_like(x_host)

    cuda.memcpy_dtoh(result, _gpu_data_buf)
    return result
