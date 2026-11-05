import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

_KERNEL_CODE = r"""
extern "C" {

__global__ void layernorm_vec4_kernel(const float* __restrict__ input,
                                      const float* __restrict__ gamma,
                                      const float* __restrict__ beta,
                                      float* __restrict__ output,
                                      int row_size,
                                      float eps) {
    __shared__ float s_sum[32];
    __shared__ float s_sq[32];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int num_warps = blockDim.x >> 5;

    const int vecs = row_size >> 2;
    const float4* in_v = reinterpret_cast<const float4*>(input + static_cast<size_t>(row) * row_size);
    float4* out_v = reinterpret_cast<float4*>(output + static_cast<size_t>(row) * row_size);
    const float4* g_v = reinterpret_cast<const float4*>(gamma);
    const float4* b_v = reinterpret_cast<const float4*>(beta);

    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (int i = tid; i < vecs; i += blockDim.x) {
        float4 v = in_v[i];
        sum += (v.x + v.y) + (v.z + v.w);
        sum_sq += (v.x * v.x + v.y * v.y) + (v.z * v.z + v.w * v.w);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, offset);
        sum_sq += __shfl_xor_sync(0xffffffffu, sum_sq, offset);
    }

    if (lane == 0) {
        s_sum[warp] = sum;
        s_sq[warp] = sum_sq;
    }
    __syncthreads();

    if (warp == 0) {
        float total_sum = (lane < num_warps) ? s_sum[lane] : 0.0f;
        float total_sq = (lane < num_warps) ? s_sq[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            total_sum += __shfl_xor_sync(0xffffffffu, total_sum, offset);
            total_sq += __shfl_xor_sync(0xffffffffu, total_sq, offset);
        }
        if (lane == 0) {
            s_sum[0] = total_sum;
            s_sq[0] = total_sq;
        }
    }
    __syncthreads();

    const float mean = s_sum[0] / row_size;
    const float variance = fmaxf(0.0f, (s_sq[0] / row_size) - mean * mean);
    const float inv_std = rsqrtf(variance + eps);

    for (int i = tid; i < vecs; i += blockDim.x) {
        float4 v = in_v[i];
        float4 g = g_v[i];
        float4 b = b_v[i];
        float4 res;
        res.x = (v.x - mean) * inv_std * g.x + b.x;
        res.y = (v.y - mean) * inv_std * g.y + b.y;
        res.z = (v.z - mean) * inv_std * g.z + b.z;
        res.w = (v.w - mean) * inv_std * g.w + b.w;
        out_v[i] = res;
    }
}

__global__ void layernorm_scalar_kernel(const float* __restrict__ input,
                                        const float* __restrict__ gamma,
                                        const float* __restrict__ beta,
                                        float* __restrict__ output,
                                        int row_size,
                                        float eps) {
    __shared__ float s_sum[32];
    __shared__ float s_sq[32];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int num_warps = blockDim.x >> 5;

    const float* in_row = input + static_cast<size_t>(row) * row_size;
    float* out_row = output + static_cast<size_t>(row) * row_size;

    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (int i = tid; i < row_size; i += blockDim.x) {
        float v = in_row[i];
        sum += v;
        sum_sq += v * v;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, offset);
        sum_sq += __shfl_xor_sync(0xffffffffu, sum_sq, offset);
    }

    if (lane == 0) {
        s_sum[warp] = sum;
        s_sq[warp] = sum_sq;
    }
    __syncthreads();

    if (warp == 0) {
        float total_sum = (lane < num_warps) ? s_sum[lane] : 0.0f;
        float total_sq = (lane < num_warps) ? s_sq[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            total_sum += __shfl_xor_sync(0xffffffffu, total_sum, offset);
            total_sq += __shfl_xor_sync(0xffffffffu, total_sq, offset);
        }
        if (lane == 0) {
            s_sum[0] = total_sum;
            s_sq[0] = total_sq;
        }
    }
    __syncthreads();

    const float mean = s_sum[0] / row_size;
    const float variance = fmaxf(0.0f, (s_sq[0] / row_size) - mean * mean);
    const float inv_std = rsqrtf(variance + eps);

    for (int i = tid; i < row_size; i += blockDim.x) {
        out_row[i] = (in_row[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

}
"""

_module = None
_kernel_vec4 = None
_kernel_scalar = None

_buffers = {
    'd_in': None,
    'd_out': None,
    'd_gamma': None,
    'd_beta': None,
    'in_cap': 0,
    'param_cap': 0,
}
_stream = None


def _init():
    global _module, _kernel_vec4, _kernel_scalar, _stream
    if _module is None:
        _module = SourceModule(_KERNEL_CODE, no_extern_c=True)
        _kernel_vec4 = _module.get_function("layernorm_vec4_kernel")
        _kernel_scalar = _module.get_function("layernorm_scalar_kernel")
        _stream = cuda.Stream()


def _ensure_buffers(total_elements, row_size):
    if _buffers['in_cap'] < total_elements:
        if _buffers['d_in'] is not None:
            _buffers['d_in'].free()
            _buffers['d_out'].free()
        _buffers['d_in'] = cuda.mem_alloc(total_elements * 4)
        _buffers['d_out'] = cuda.mem_alloc(total_elements * 4)
        _buffers['in_cap'] = total_elements

    if _buffers['param_cap'] < row_size:
        if _buffers['d_gamma'] is not None:
            _buffers['d_gamma'].free()
            _buffers['d_beta'].free()
        _buffers['d_gamma'] = cuda.mem_alloc(row_size * 4)
        _buffers['d_beta'] = cuda.mem_alloc(row_size * 4)
        _buffers['param_cap'] = row_size


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    """
    Apply Layer Normalization to each row of the input matrix.

    Parameters
    ----------
    input : list or numpy.ndarray of float
        Flattened matrix in row‑major order. Its length must be divisible by row_size.
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
        Flattened matrix of the same shape as input, containing the row‑wise
        normalized results.
    """
    input_arr = np.ascontiguousarray(input, dtype=np.float32).ravel()
    gamma_arr = np.ascontiguousarray(gamma, dtype=np.float32).ravel()
    beta_arr = np.ascontiguousarray(beta, dtype=np.float32).ravel()

    total_elements = input_arr.size
    if total_elements == 0:
        return np.empty(0, dtype=np.float32)

    row_size = int(row_size)
    row_count = total_elements // row_size

    _init()
    _ensure_buffers(total_elements, row_size)

    d_in = _buffers['d_in']
    d_out = _buffers['d_out']
    d_gamma = _buffers['d_gamma']
    d_beta = _buffers['d_beta']

    cuda.memcpy_htod_async(d_in, input_arr, _stream)
    cuda.memcpy_htod_async(d_gamma, gamma_arr, _stream)
    cuda.memcpy_htod_async(d_beta, beta_arr, _stream)

    block_size = 256

    if row_size % 4 == 0:
        _kernel_vec4(
            d_in, d_gamma, d_beta, d_out,
            np.int32(row_size), np.float32(eps),
            block=(block_size, 1, 1),
            grid=(row_count, 1, 1),
            stream=_stream
        )
    else:
        _kernel_scalar(
            d_in, d_gamma, d_beta, d_out,
            np.int32(row_size), np.float32(eps),
            block=(block_size, 1, 1),
            grid=(row_count, 1, 1),
            stream=_stream
        )

    output = np.empty_like(input_arr)
    cuda.memcpy_dtoh_async(output, d_out, _stream)
    _stream.synchronize()

    return output
