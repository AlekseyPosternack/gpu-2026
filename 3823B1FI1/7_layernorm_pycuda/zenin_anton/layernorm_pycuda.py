import numpy as np
import pycuda.autoinit  
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

_KERNEL_TEMPLATE = r"""
#define ROW_SIZE {row_size}
#define BLOCK {block}
#define ITEMS {items}
#define WARP_SIZE 32
#define WARPS (BLOCK / WARP_SIZE)

__device__ __forceinline__ float WarpSum(float value) {{
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {{
        value += __shfl_xor_sync(0xffffffff, value, offset);
    }}
    return value;
}}

__device__ __forceinline__ float BlockSum(float value, float* warp_sums) {{
    value = WarpSum(value);
#if BLOCK > WARP_SIZE
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    if (lane == 0) {{
        warp_sums[warp] = value;
    }}
    __syncthreads();
    value = (lane < WARPS) ? warp_sums[lane] : 0.0f;
    value = WarpSum(value);   
    __syncthreads();          
#endif
    return value;
}}

__global__ void __launch_bounds__(BLOCK)
layernorm_kernel(float* data,
                 const float* __restrict__ gamma,
                 const float* __restrict__ beta,
                 float eps) {{
    __shared__ float warp_sums[WARPS];
    float* row = data + (size_t)blockIdx.x * ROW_SIZE;
    const float inv_n = 1.0f / ROW_SIZE;

#if ITEMS > 0
    
    float values[ITEMS];
    float local_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < ITEMS; ++i) {{
        const int j = threadIdx.x + i * BLOCK;
        values[i] = (j < ROW_SIZE) ? row[j] : 0.0f;
        local_sum += values[i];
    }}
    const float mean = BlockSum(local_sum, warp_sums) * inv_n;

    float local_sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < ITEMS; ++i) {{
        const int j = threadIdx.x + i * BLOCK;
        const float d = values[i] - mean;
        local_sq += (j < ROW_SIZE) ? d * d : 0.0f;
    }}
    const float var = BlockSum(local_sq, warp_sums) * inv_n;
    const float rstd = rsqrtf(var + eps);   // 1 / sqrt(var + eps)

    // 3. Normalize, scale, shift and write
    #pragma unroll
    for (int i = 0; i < ITEMS; ++i) {{
        const int j = threadIdx.x + i * BLOCK;
        if (j < ROW_SIZE) {{
            row[j] = (values[i] - mean) * rstd * gamma[j] + beta[j];
        }}
    }}
#else
    
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < ROW_SIZE; j += BLOCK) {{
        local_sum += row[j];
    }}
    const float mean = BlockSum(local_sum, warp_sums) * inv_n;

    float local_sq = 0.0f;
    for (int j = threadIdx.x; j < ROW_SIZE; j += BLOCK) {{
        const float d = row[j] - mean;
        local_sq += d * d;
    }}
    const float var = BlockSum(local_sq, warp_sums) * inv_n;
    const float rstd = rsqrtf(var + eps);

    for (int j = threadIdx.x; j < ROW_SIZE; j += BLOCK) {{
        row[j] = (row[j] - mean) * rstd * gamma[j] + beta[j];
    }}
#endif
}}
"""

_MAX_CACHED_ROW = 16384   

_kernels = {}             
_buffers = {}             


def _launch_config(row_size):
    if row_size <= 512:
        block = 32                                   
        items = max(1, (row_size + block - 1) // block)
    elif row_size <= _MAX_CACHED_ROW:
        block = 512
        items = (row_size + block - 1) // block      
    else:
        block = 512
        items = 0                                    
    return block, items


def _make_source(row_size):
    block, items = _launch_config(row_size)
    source = _KERNEL_TEMPLATE.format(row_size=row_size, block=block, items=items)
    return source, block


def _get_kernel(row_size):
    if row_size not in _kernels:
        source, block = _make_source(row_size)
        module = SourceModule(source, options=["-O3", "--use_fast_math"])
        _kernels[row_size] = (module.get_function("layernorm_kernel"), block)
    return _kernels[row_size]


def _get_buffer(name, nbytes):
    buffer = _buffers.get(name)
    if buffer is None or buffer[1] < nbytes:
        if buffer is not None:
            buffer[0].free()
        buffer = (cuda.mem_alloc(nbytes), nbytes)
        _buffers[name] = buffer
    return buffer[0]


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
    
    x = np.ascontiguousarray(input, dtype=np.float32).ravel()
    g = np.ascontiguousarray(gamma, dtype=np.float32).ravel()
    b = np.ascontiguousarray(beta, dtype=np.float32).ravel()

    row_size = int(row_size)
    if x.size == 0:
        return np.empty(0, dtype=np.float32)
    row_count = x.size // row_size

    kernel, block = _get_kernel(row_size)
    d_data = _get_buffer("data", x.nbytes)
    d_gamma = _get_buffer("gamma", g.nbytes)
    d_beta = _get_buffer("beta", b.nbytes)

    cuda.memcpy_htod(d_data, x)
    cuda.memcpy_htod(d_gamma, g)
    cuda.memcpy_htod(d_beta, b)

    
    output = np.empty_like(x)

    kernel(d_data, d_gamma, d_beta, np.float32(eps),
           block=(block, 1, 1), grid=(row_count, 1))

    
    cuda.memcpy_dtoh(output, d_data)
    return output
