import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

_kernel_code = r"""
__global__ void layernorm_kernel(const float* __restrict__ input,
                                 const float* __restrict__ gamma,
                                 const float* __restrict__ beta,
                                 float* __restrict__ output,
                                 int row_size,
                                 float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int offset = row * row_size;

    extern __shared__ float sdata[];

    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (int i = tid; i < row_size; i += blockDim.x) {
        float val = input[offset + i];
        sum += val;
        sum_sq += val * val;
    }

    sdata[tid] = sum;
    sdata[blockDim.x + tid] = sum_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
            sdata[blockDim.x + tid] += sdata[blockDim.x + tid + stride];
        }
        __syncthreads();
    }

    const float mean = sdata[0] / row_size;
    const float variance = (sdata[blockDim.x] / row_size) - (mean * mean);
    const float inv_std = rsqrtf(variance + eps);

    for (int i = tid; i < row_size; i += blockDim.x) {
        float val = input[offset + i];
        float normalized = (val - mean) * inv_std;
        output[offset + i] = normalized * gamma[i] + beta[i];
    }
}
"""

_mod = SourceModule(_kernel_code)
_kernel = _mod.get_function("layernorm_kernel")

_cache = {
    'd_x': None, 'd_g': None, 'd_b': None, 'd_y': None,
    'cap_x': 0, 'cap_g': 0
}
_stream = None

def _ensure_memory(total_elements, row_size):
    """Выделяет память на GPU, только если её еще нет или она мала."""
    global _stream
    if _stream is None:
        _stream = drv.Stream()

    if _cache['cap_x'] < total_elements:
        if _cache['d_x'] is not None:
            _cache['d_x'].free()
            _cache['d_y'].free()
        _cache['d_x'] = drv.mem_alloc(total_elements * 4)
        _cache['d_y'] = drv.mem_alloc(total_elements * 4)
        _cache['cap_x'] = total_elements

    if _cache['cap_g'] < row_size:
        if _cache['d_g'] is not None:
            _cache['d_g'].free()
            _cache['d_b'].free()
        _cache['d_g'] = drv.mem_alloc(row_size * 4)
        _cache['d_b'] = drv.mem_alloc(row_size * 4)
        _cache['cap_g'] = row_size


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    """
    Apply Layer Normalization to each row of the input matrix.
    """
    input_np = np.asarray(input, dtype=np.float32).ravel()
    gamma_np = np.asarray(gamma, dtype=np.float32).ravel()
    beta_np = np.asarray(beta, dtype=np.float32).ravel()

    total_elements = input_np.size
    if total_elements == 0:
        return np.array([], dtype=np.float32)

    if total_elements % row_size != 0:
        raise ValueError("input size must be divisible by row_size")

    row_count = total_elements // row_size
    output_np = np.empty_like(input_np)

    _ensure_memory(total_elements, row_size)
    d_x, d_g, d_b, d_y = _cache['d_x'], _cache['d_g'], _cache['d_b'], _cache['d_y']

    drv.memcpy_htod_async(d_x, input_np, _stream)
    drv.memcpy_htod_async(d_g, gamma_np, _stream)
    drv.memcpy_htod_async(d_b, beta_np, _stream)

    block_size = 256
    shared_mem_bytes = 2 * block_size * 4

    _kernel(
        d_x, d_g, d_b, d_y,
        np.int32(row_size),
        np.float32(eps),
        block=(block_size, 1, 1),
        grid=(row_count, 1),
        shared=shared_mem_bytes,
        stream=_stream
    )

    drv.memcpy_dtoh_async(output_np, d_y, _stream)
    
    _stream.synchronize()

    return output_np