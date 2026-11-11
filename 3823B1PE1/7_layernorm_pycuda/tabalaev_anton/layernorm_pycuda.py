import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

_KERNEL_SOURCE = """
#define BLOCK_SIZE 256

__global__ void layernorm_row_kernel(float* __restrict__ data,
                                     const float* __restrict__ gamma,
                                     const float* __restrict__ beta,
                                     int row_size,
                                     float eps) {
    __shared__ float s_reduce[BLOCK_SIZE];

    const int tid = threadIdx.x;
    float* row_ptr = data + static_cast<size_t>(blockIdx.x) * row_size;

    const int vec_cols = row_size / 4;
    float4* row_vec = reinterpret_cast<float4*>(row_ptr);

    float local_sum = 0.0f;
    for (int i = tid; i < vec_cols; i += BLOCK_SIZE) {
        const float4 v = row_vec[i];
        local_sum += (v.x + v.y) + (v.z + v.w);
    }
    for (int i = vec_cols * 4 + tid; i < row_size; i += BLOCK_SIZE) {
        local_sum += row_ptr[i];
    }

    s_reduce[tid] = local_sum;
    __syncthreads();

    #pragma unroll
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_reduce[tid] += s_reduce[tid + stride];
        }
        __syncthreads();
    }
    const float mean = s_reduce[0] / static_cast<float>(row_size);

    float local_var = 0.0f;
    for (int i = tid; i < vec_cols; i += BLOCK_SIZE) {
        const float4 v = row_vec[i];
        const float dx = v.x - mean;
        const float dy = v.y - mean;
        const float dz = v.z - mean;
        const float dw = v.w - mean;
        local_var += (dx * dx + dy * dy) + (dz * dz + dw * dw);
    }
    for (int i = vec_cols * 4 + tid; i < row_size; i += BLOCK_SIZE) {
        const float diff = row_ptr[i] - mean;
        local_var += diff * diff;
    }

    s_reduce[tid] = local_var;
    __syncthreads();

    #pragma unroll
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_reduce[tid] += s_reduce[tid + stride];
        }
        __syncthreads();
    }
    const float inv_std = rsqrtf(s_reduce[0] / static_cast<float>(row_size) + eps);

    const float4* gamma_vec = reinterpret_cast<const float4*>(gamma);
    const float4* beta_vec = reinterpret_cast<const float4*>(beta);

    for (int i = tid; i < vec_cols; i += BLOCK_SIZE) {
        float4 v = row_vec[i];
        const float4 g = gamma_vec[i];
        const float4 b = beta_vec[i];
        v.x = g.x * (v.x - mean) * inv_std + b.x;
        v.y = g.y * (v.y - mean) * inv_std + b.y;
        v.z = g.z * (v.z - mean) * inv_std + b.z;
        v.w = g.w * (v.w - mean) * inv_std + b.w;
        row_vec[i] = v;
    }
    for (int i = vec_cols * 4 + tid; i < row_size; i += BLOCK_SIZE) {
        row_ptr[i] = gamma[i] * (row_ptr[i] - mean) * inv_std + beta[i];
    }
}
"""

_module = SourceModule(_KERNEL_SOURCE)
_layernorm_kernel = _module.get_function("layernorm_row_kernel")
_stream = cuda.Stream()

_d_data = None
_d_gamma = None
_d_beta = None
_data_capacity = 0
_param_capacity = 0


def _ensure_gpu_buffers(data_bytes, param_bytes):
    global _d_data, _d_gamma, _d_beta, _data_capacity, _param_capacity

    if data_bytes > _data_capacity:
        if _d_data is not None:
            _d_data.free()
        _d_data = cuda.mem_alloc(data_bytes)
        _data_capacity = data_bytes

    if param_bytes > _param_capacity:
        if _d_gamma is not None:
            _d_gamma.free()
        if _d_beta is not None:
            _d_beta.free()
        _d_gamma = cuda.mem_alloc(param_bytes)
        _d_beta = cuda.mem_alloc(param_bytes)
        _param_capacity = param_bytes


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
    in_arr = np.ascontiguousarray(input, dtype=np.float32)
    total_elements = in_arr.size
    if total_elements == 0 or row_size <= 0:
        return np.empty_like(in_arr)

    gamma_arr = np.ascontiguousarray(gamma, dtype=np.float32)
    beta_arr = np.ascontiguousarray(beta, dtype=np.float32)

    _ensure_gpu_buffers(in_arr.nbytes, gamma_arr.nbytes)

    cuda.memcpy_htod_async(_d_data, in_arr, _stream)
    cuda.memcpy_htod_async(_d_gamma, gamma_arr, _stream)
    cuda.memcpy_htod_async(_d_beta, beta_arr, _stream)

    row_count = int(total_elements // row_size)
    _layernorm_kernel(
        _d_data,
        _d_gamma,
        _d_beta,
        np.int32(row_size),
        np.float32(eps),
        block=(256, 1, 1),
        grid=(row_count, 1, 1),
        stream=_stream,
    )

    output = np.empty_like(in_arr)

    cuda.memcpy_dtoh_async(output, _d_data, _stream)
    _stream.synchronize()

    return output
