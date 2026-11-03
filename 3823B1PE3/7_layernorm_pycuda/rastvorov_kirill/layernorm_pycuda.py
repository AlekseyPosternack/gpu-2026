import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

_BLOCK = 256

_KERNEL_SOURCE = r"""
#define BLOCK 256
#define WARPS (BLOCK / 32)

//------------------------------------------------------------------
//function: warp_sum
//purpose: складывает значения 32 потоков варпа через обмен регистрами без общей памяти
//------------------------------------------------------------------
__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, offset);
    return v;
}

//------------------------------------------------------------------
//function: block_sum
//purpose: складывает значения всех потоков блока и раздаёт сумму каждому потоку
//------------------------------------------------------------------
__device__ __forceinline__ float block_sum(float v, float* shared) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    v = warp_sum(v);
    if (lane == 0) shared[warp] = v;
    __syncthreads();
    v = lane < WARPS ? shared[lane] : 0.0f;
    v = warp_sum(v);
    __syncthreads();
    return v;
}

//------------------------------------------------------------------
//function: layernorm
//purpose: нормирует одну строку на блок: находит среднее и дисперсию, затем применяет gamma и beta
//------------------------------------------------------------------
extern "C" __global__ void layernorm(const float* __restrict__ in,
                                     const float* __restrict__ gamma,
                                     const float* __restrict__ beta,
                                     float* __restrict__ out,
                                     int row_size, float eps) {
    __shared__ float shared[WARPS];
    const float* row_in = in + (size_t)blockIdx.x * row_size;
    float* row_out = out + (size_t)blockIdx.x * row_size;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += BLOCK) sum += row_in[i];
    const float mean = block_sum(sum, shared) / row_size;

    float sq = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += BLOCK) {
        const float d = row_in[i] - mean;
        sq += d * d;
    }
    const float inv_std = rsqrtf(block_sum(sq, shared) / row_size + eps);

    for (int i = threadIdx.x; i < row_size; i += BLOCK)
        row_out[i] = (row_in[i] - mean) * inv_std * gamma[i] + beta[i];
}
"""

_kernel = None
_buffers = {}


#------------------------------------------------------------------
#function: _get_kernel
#purpose: компилирует CUDA-ядро один раз при первом вызове и дальше берёт готовое
#------------------------------------------------------------------
def _get_kernel():
    global _kernel
    if _kernel is None:
        module = SourceModule(_KERNEL_SOURCE, no_extern_c=True,
                              options=["-O3", "--use_fast_math"])
        _kernel = module.get_function("layernorm")
        _kernel.prepare("PPPPif")
    return _kernel


#------------------------------------------------------------------
#function: _get_buffer
#purpose: выдаёт буфер видеопамяти нужного размера, выделяя новый только если старый мал
#------------------------------------------------------------------
def _get_buffer(name, nbytes):
    buf = _buffers.get(name)
    if buf is None or buf[1] < nbytes:
        buf = (drv.mem_alloc(nbytes), nbytes)
        _buffers[name] = buf
    return buf[0]


#------------------------------------------------------------------
#function: layernorm_pycuda
#purpose: применяет LayerNorm к каждой строке матрицы на GPU и возвращает результат в float32
#------------------------------------------------------------------
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
    x = np.ascontiguousarray(input, dtype=np.float32).ravel()
    g = np.ascontiguousarray(gamma, dtype=np.float32).ravel()
    b = np.ascontiguousarray(beta, dtype=np.float32).ravel()
    row_size = int(row_size)
    if x.size == 0 or row_size <= 0:
        return np.empty(0, dtype=np.float32)
    row_count = x.size // row_size

    kernel = _get_kernel()
    d_in = _get_buffer("in", x.nbytes)
    d_out = _get_buffer("out", x.nbytes)
    d_gamma = _get_buffer("gamma", g.nbytes)
    d_beta = _get_buffer("beta", b.nbytes)

    drv.memcpy_htod(d_in, x)
    drv.memcpy_htod(d_gamma, g)
    drv.memcpy_htod(d_beta, b)

    kernel.prepared_call((row_count, 1, 1), (_BLOCK, 1, 1),
                         d_in, d_gamma, d_beta, d_out,
                         np.int32(row_size), np.float32(eps))

    result = np.empty_like(x)
    drv.memcpy_dtoh(result, d_out)
    return result
