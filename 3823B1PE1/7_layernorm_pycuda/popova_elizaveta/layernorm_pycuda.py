import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule, compile as cuda_compile

_SRC = r"""
#define THREADS 256
#define NWARPS  (THREADS / 32)

__device__ __forceinline__ float warpReduceSum(float v) {
    #pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

__global__ void layernorm_kernel(const float* __restrict__ in,
                                 const float* __restrict__ gamma,
                                 const float* __restrict__ beta,
                                 float* __restrict__ out,
                                 int row_size, float eps) {
    __shared__ float s_warp[NWARPS];
    const int row = blockIdx.x, tid = threadIdx.x;
    const int warp = tid / 32, lane = tid % 32;
    const float* row_in  = in  + (size_t)row * row_size;
    float*       row_out = out + (size_t)row * row_size;

    float s = 0.f;
    for (int i = tid; i < row_size; i += THREADS) s += row_in[i];
    s = warpReduceSum(s);
    if (!lane) s_warp[warp] = s;
    __syncthreads();
    float sum = s_warp[0];
    #pragma unroll
    for (int i = 1; i < NWARPS; ++i) sum += s_warp[i];
    const float mean = sum / (float)row_size;

    float v = 0.f;
    for (int i = tid; i < row_size; i += THREADS) {
        float d = row_in[i] - mean;
        v += d * d;
    }
    v = warpReduceSum(v);
    if (!lane) s_warp[warp] = v;
    __syncthreads();
    float var = s_warp[0];
    #pragma unroll
    for (int i = 1; i < NWARPS; ++i) var += s_warp[i];
    const float inv_std = rsqrtf(var / (float)row_size + eps);

    for (int i = tid; i < row_size; i += THREADS)
        row_out[i] = gamma[i] * (row_in[i] - mean) * inv_std + beta[i];
}
"""

try:
    _mod = SourceModule(_SRC)
except Exception:
    _mod = drv.module_from_buffer(
        cuda_compile(_SRC, target="ptx", arch="compute_89", options=["-O3"])
    )
_kernel = _mod.get_function("layernorm_kernel")

_cache = {}


def _buf(key, nbytes):
    b = _cache.get(key)
    if b is None or b[1] < nbytes:
        if b is not None:
            b[0].free()
        _cache[key] = (drv.mem_alloc(nbytes), nbytes)
    return _cache[key][0]


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    x = np.ascontiguousarray(input, dtype=np.float32).ravel()
    g = np.ascontiguousarray(gamma, dtype=np.float32).ravel()
    b = np.ascontiguousarray(beta, dtype=np.float32).ravel()

    if x.size == 0 or row_size <= 0 or x.size % row_size != 0:
        raise ValueError("invalid input/row_size")

    rows = x.size // row_size
    out = np.empty_like(x)

    d_x = _buf("x", x.nbytes)
    d_g = _buf("g", g.nbytes)
    d_b = _buf("b", b.nbytes)
    d_y = _buf("y", out.nbytes)

    drv.memcpy_htod(d_x, x)
    drv.memcpy_htod(d_g, g)
    drv.memcpy_htod(d_b, b)

    _kernel(
        d_x,
        d_g,
        d_b,
        d_y,
        np.int32(row_size),
        np.float32(eps),
        block=(256, 1, 1),
        grid=(rows, 1, 1),
    )

    drv.memcpy_dtoh(out, d_y)
    return out
