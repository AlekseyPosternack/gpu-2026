import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as drv
from pycuda.compiler import SourceModule


_kernel_code = r"""
#define MAX_VEC_PER_THREAD 16

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}


__global__ void layernorm_kernel_vec(
        const float* __restrict__ x,
        const float* __restrict__ gamma,
        const float* __restrict__ beta,
        float* __restrict__ y,
        int row_size,
        float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const float* row_x = x + (size_t)row * row_size;
    float*       row_y = y + (size_t)row * row_size;

    const int vec_per_thread = row_size / (nthreads * 4);
    float4 vals[MAX_VEC_PER_THREAD];

    float local_sum = 0.0f;
    for (int i = 0; i < vec_per_thread; ++i) {
        int vec_idx = i * nthreads + tid;
        float4 v = *reinterpret_cast<const float4*>(
            row_x + vec_idx * 4);
        vals[i] = v;
        local_sum += (v.x + v.y) + (v.z + v.w);
    }

    __shared__ float red[32];
    const int lane = tid & 31;
    const int wid  = tid >> 5;
    const int nwarps = nthreads >> 5;

    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) red[wid] = local_sum;
    __syncthreads();

    float total = 0.0f;
    #pragma unroll
    for (int i = 0; i < nwarps; ++i) total += red[i];

    const float mean = total / row_size;

    float local_var = 0.0f;
    for (int i = 0; i < vec_per_thread; ++i) {
        float4 v = vals[i];
        float dx = v.x - mean;
        float dy = v.y - mean;
        float dz = v.z - mean;
        float dw = v.w - mean;
        local_var += dx*dx + dy*dy + dz*dz + dw*dw;
    }

    local_var = warp_reduce_sum(local_var);
    if (lane == 0) red[wid] = local_var;
    __syncthreads();

    float var_total = 0.0f;
    #pragma unroll
    for (int i = 0; i < nwarps; ++i) var_total += red[i];

    const float inv_std = rsqrtf(var_total / row_size + eps);

    for (int i = 0; i < vec_per_thread; ++i) {
        int vec_idx = i * nthreads + tid;
        float4 v = vals[i];
        float4 g = *reinterpret_cast<const float4*>(
            gamma + vec_idx * 4);
        float4 b = *reinterpret_cast<const float4*>(
            beta  + vec_idx * 4);

        v.x = (v.x - mean) * inv_std * g.x + b.x;
        v.y = (v.y - mean) * inv_std * g.y + b.y;
        v.z = (v.z - mean) * inv_std * g.z + b.z;
        v.w = (v.w - mean) * inv_std * g.w + b.w;

        *reinterpret_cast<float4*>(row_y + vec_idx * 4) = v;
    }
}

__global__ void layernorm_kernel_scalar(
        const float* __restrict__ x,
        const float* __restrict__ gamma,
        const float* __restrict__ beta,
        float* __restrict__ y,
        int row_size,
        float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const float* row_x = x + (size_t)row * row_size;
    float*       row_y = y + (size_t)row * row_size;

    float local_sum = 0.0f;
    for (int i = tid; i < row_size; i += nthreads) {
        local_sum += row_x[i];
    }

    __shared__ float red[32];
    const int lane = tid & 31;
    const int wid  = tid >> 5;
    const int nwarps = nthreads >> 5;

    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) red[wid] = local_sum;
    __syncthreads();

    float total = 0.0f;
    #pragma unroll
    for (int i = 0; i < nwarps; ++i) total += red[i];

    const float mean = total / row_size;

    float local_var = 0.0f;
    for (int i = tid; i < row_size; i += nthreads) {
        float d = row_x[i] - mean;
        local_var += d * d;
    }

    local_var = warp_reduce_sum(local_var);
    if (lane == 0) red[wid] = local_var;
    __syncthreads();

    float var_total = 0.0f;
    #pragma unroll
    for (int i = 0; i < nwarps; ++i) var_total += red[i];

    const float inv_std = rsqrtf(var_total / row_size + eps);

    for (int i = tid; i < row_size; i += nthreads) {
        row_y[i] = (row_x[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}
"""

_mod = SourceModule(_kernel_code)
_layernorm_kernel_vec = _mod.get_function("layernorm_kernel_vec")
_layernorm_kernel_scalar = _mod.get_function("layernorm_kernel_scalar")


def _next_pow2_le(n):
    """Наибольшая степень двойки <= n."""
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


_d_x = None
_d_g = None
_d_b = None
_d_y = None
_cap_x = 0
_cap_g = 0

_h_x = None
_h_y = None
_cap_h = 0

_stream = None


def _ensure_buffers(elems, row_size):
    global _d_x, _d_y, _cap_x
    global _d_g, _d_b, _cap_g
    global _h_x, _h_y, _cap_h
    global _stream

    if _stream is None:
        _stream = drv.Stream()

    if _cap_x < elems:
        if _d_x is not None:
            _d_x.free()
        if _d_y is not None:
            _d_y.free()
        _d_x = drv.mem_alloc(elems * 4)
        _d_y = drv.mem_alloc(elems * 4)
        _cap_x = elems

    if _cap_g < row_size:
        if _d_g is not None:
            _d_g.free()
        if _d_b is not None:
            _d_b.free()
        _d_g = drv.mem_alloc(row_size * 4)
        _d_b = drv.mem_alloc(row_size * 4)
        _cap_g = row_size

    if _cap_h < elems:
        if _h_x is not None:
            _h_x.free()
        if _h_y is not None:
            _h_y.free()
        _h_x = drv.pagelocked_empty(elems, np.float32)
        _h_y = drv.pagelocked_empty(elems, np.float32)
        _cap_h = elems


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):

    x = np.asarray(input, dtype=np.float32)
    g = np.asarray(gamma, dtype=np.float32)
    b = np.asarray(beta, dtype=np.float32)

    total = x.size
    if total == 0:
        return np.empty_like(x)

    assert total % row_size == 0, (
        "input size must be divisible by row_size"
    )
    assert g.size == row_size and b.size == row_size

    row_count = total // row_size
    y = np.empty_like(x)

    _ensure_buffers(total, row_size)

    _h_x[:total] = x.ravel()
    drv.memcpy_htod_async(_d_x, _h_x, _stream)
    drv.memcpy_htod_async(_d_g, g, _stream)
    drv.memcpy_htod_async(_d_b, b, _stream)

    use_vec = (
        row_size >= 4
        and row_size % 4 == 0
        and row_size <= 16 * 512 * 4   # MAX_VEC_PER_THREAD * max_block * 4
    )

    if use_vec:
        block_threads = _next_pow2_le(min(512, row_size // 4))
        while block_threads > 1 and row_size % (block_threads * 4) != 0:
            block_threads //= 2
        vec_per_thread = row_size // (block_threads * 4)
        assert vec_per_thread <= 16, "vec_per_thread exceeds MAX_VEC"

        _layernorm_kernel_vec(
            _d_x, _d_g, _d_b, _d_y,
            np.int32(row_size),
            np.float32(eps),
            grid=(row_count, 1, 1),
            block=(block_threads, 1, 1),
            stream=_stream,
        )
    else:
        block_threads = _next_pow2_le(min(256, max(32, row_size)))
        if block_threads < 32:
            block_threads = 32

        _layernorm_kernel_scalar(
            _d_x, _d_g, _d_b, _d_y,
            np.int32(row_size),
            np.float32(eps),
            grid=(row_count, 1, 1),
            block=(block_threads, 1, 1),
            stream=_stream,
        )

    drv.memcpy_dtoh_async(_h_y, _d_y, _stream)
    _stream.synchronize()

    y[:] = _h_y[:total]

    return y