import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

_KERNEL = r"""
#ifndef CACHE
#define CACHE 0
#endif

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ float block_sum(float v, float* red) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    v = (lane < (blockDim.x >> 5)) ? red[lane] : 0.f;
    return warp_sum(v);
}

extern "C" __global__ void layernorm(const float* __restrict__ in,
                                     float* __restrict__ out,
                                     const float* __restrict__ gamma,
                                     const float* __restrict__ beta,
                                     int row_size, float eps) {
    __shared__ float red[32];

    const float* x = in + (size_t)blockIdx.x * row_size;
    float* y = out + (size_t)blockIdx.x * row_size;

#if CACHE == 2
    const int tid = threadIdx.x;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float4* y4 = reinterpret_cast<float4*>(y);
    const float4* g4 = reinterpret_cast<const float4*>(gamma);
    const float4* b4 = reinterpret_cast<const float4*>(beta);

    float4 reg[8];
    float s = 0.f;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        float4 v = x4[tid + k * 512];
        reg[k] = v;
        s += (v.x + v.y) + (v.z + v.w);
    }
    const float mean = block_sum(s, red) * (1.0f / 16384.0f);

    float q = 0.f;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        reg[k].x -= mean;
        reg[k].y -= mean;
        reg[k].z -= mean;
        reg[k].w -= mean;
        q += (reg[k].x * reg[k].x + reg[k].y * reg[k].y) +
             (reg[k].z * reg[k].z + reg[k].w * reg[k].w);
    }
    const float inv_std = rsqrtf(block_sum(q, red) * (1.0f / 16384.0f) + eps);

    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        int idx = tid + k * 512;
        float4 gv = g4[idx];
        float4 bv = b4[idx];
        float4 res;
        res.x = reg[k].x * inv_std * gv.x + bv.x;
        res.y = reg[k].y * inv_std * gv.y + bv.y;
        res.z = reg[k].z * inv_std * gv.z + bv.z;
        res.w = reg[k].w * inv_std * gv.w + bv.w;
        y4[idx] = res;
    }
#else
    extern __shared__ float row[];

    float s = 0.f;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        const float v = x[i];
#if CACHE == 1
        row[i] = v;
#endif
        s += v;
    }
    const float mean = block_sum(s, red) / row_size;

    float q = 0.f;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
#if CACHE == 1
        const float d = row[i] - mean;
#else
        const float d = x[i] - mean;
#endif
        q += d * d;
    }
    const float inv_std = rsqrtf(block_sum(q, red) / row_size + eps);

    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
#if CACHE == 1
        const float v = row[i];
#else
        const float v = x[i];
#endif
        y[i] = (v - mean) * inv_std * gamma[i] + beta[i];
    }
#endif
}
"""

_MAX_CACHE_FLOATS = 8192

_funcs = {}
_bufs = {"cap": 0, "in": None, "out": None}
_params = {"cap": 0, "gamma": None, "beta": None}


def _get_func(cache):
    f = _funcs.get(cache)
    if f is None:
        mod = SourceModule(_KERNEL, options=["-DCACHE=%d" % cache])
        f = mod.get_function("layernorm")
        _funcs[cache] = f
    return f


def _reserve(nbytes, row_bytes):
    if nbytes > _bufs["cap"]:
        if _bufs["in"] is not None:
            _bufs["in"].free()
            _bufs["out"].free()
        _bufs["in"] = cuda.mem_alloc(nbytes)
        _bufs["out"] = cuda.mem_alloc(nbytes)
        _bufs["cap"] = nbytes
    if row_bytes > _params["cap"]:
        if _params["gamma"] is not None:
            _params["gamma"].free()
            _params["beta"].free()
        _params["gamma"] = cuda.mem_alloc(row_bytes)
        _params["beta"] = cuda.mem_alloc(row_bytes)
        _params["cap"] = row_bytes


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    x = np.ascontiguousarray(input, dtype=np.float32).ravel()
    g = np.ascontiguousarray(gamma, dtype=np.float32).ravel()
    b = np.ascontiguousarray(beta, dtype=np.float32).ravel()
    row_size = int(row_size)

    if x.size == 0 or row_size <= 0:
        return np.empty(0, dtype=np.float32)
    rows = x.size // row_size

    _reserve(x.nbytes, g.nbytes)
    cuda.memcpy_htod(_bufs["in"], x)
    cuda.memcpy_htod(_params["gamma"], g)
    cuda.memcpy_htod(_params["beta"], b)

    if row_size == 16384:
        block = 512
        cache = 2
        shared_bytes = 0
    else:
        block = 32
        while block < 512 and block < row_size:
            block <<= 1
        cache = 1 if row_size <= _MAX_CACHE_FLOATS else 0
        shared_bytes = row_size * 4 if cache == 1 else 0

    func = _get_func(cache)
    func(_bufs["in"], _bufs["out"], _params["gamma"], _params["beta"],
         np.int32(row_size), np.float32(eps),
         block=(block, 1, 1), grid=(rows, 1),
         shared=shared_bytes)

    out = np.empty_like(x)
    cuda.memcpy_dtoh(out, _bufs["out"])
    return out