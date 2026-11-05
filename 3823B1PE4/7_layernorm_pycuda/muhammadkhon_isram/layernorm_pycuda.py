import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

_KERNEL_SOURCE = r"""
#define WARP_SIZE 32
#define MAX_BLOCK_SIZE 256

__device__ __forceinline__ float WarpSum(float v) {
    #pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

// Сумма по блоку, результат у всех потоков
__device__ float BlockSum(float v, float* red) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    v = WarpSum(v);
    __syncthreads();                       // red мог использоваться прошлой редукцией
    if (lane == 0) red[warp] = v;
    __syncthreads();
    v = lane < blockDim.x / WARP_SIZE ? red[lane] : 0.0f;
    return WarpSum(v);                     // все варпы считают одно и то же
}

// Один блок = одна строка. cached != 0: строка целиком в shared memory.
// vec4 != 0: row_size % 4 == 0, работаем через float4.
__global__ void LayerNorm(const float* __restrict__ in, const float* __restrict__ gamma,
                          const float* __restrict__ beta, float* __restrict__ out,
                          int row_size, float eps, int cached, int vec4) {
    extern __shared__ float cache[];
    __shared__ float red[MAX_BLOCK_SIZE / WARP_SIZE];

    const size_t base = (size_t)blockIdx.x * row_size;
    const float* row_in = in + base;
    float* row_out = out + base;
    const int tid = threadIdx.x;

    // Проход 1: сумма (единственное чтение строки из глобальной памяти)
    float sum = 0.0f;
    if (vec4) {
        const float4* in4 = reinterpret_cast<const float4*>(row_in);
        for (int i = tid; i < row_size / 4; i += blockDim.x) {
            const float4 v = in4[i];
            if (cached) reinterpret_cast<float4*>(cache)[i] = v;
            sum += (v.x + v.y) + (v.z + v.w);
        }
    } else {
        for (int i = tid; i < row_size; i += blockDim.x) {
            const float v = row_in[i];
            if (cached) cache[i] = v;
            sum += v;
        }
    }
    const float mean = BlockSum(sum, red) / row_size;

    // Проход 2: двухпроходная дисперсия (данные в shared или L2)
    float sq = 0.0f;
    if (vec4) {
        for (int i = tid; i < row_size / 4; i += blockDim.x) {
            const float4 v = cached ? reinterpret_cast<const float4*>(cache)[i]
                                    : reinterpret_cast<const float4*>(row_in)[i];
            const float a = v.x - mean, b = v.y - mean, c = v.z - mean, d = v.w - mean;
            sq += (a * a + b * b) + (c * c + d * d);
        }
    } else {
        for (int i = tid; i < row_size; i += blockDim.x) {
            const float d = (cached ? cache[i] : row_in[i]) - mean;
            sq += d * d;
        }
    }
    const float inv_std = rsqrtf(BlockSum(sq, red) / row_size + eps);

    // Проход 3: нормировка
    if (vec4) {
        const float4* g4 = reinterpret_cast<const float4*>(gamma);
        const float4* b4 = reinterpret_cast<const float4*>(beta);
        float4* out4 = reinterpret_cast<float4*>(row_out);
        for (int i = tid; i < row_size / 4; i += blockDim.x) {
            const float4 v = cached ? reinterpret_cast<const float4*>(cache)[i]
                                    : reinterpret_cast<const float4*>(row_in)[i];
            const float4 g = g4[i], b = b4[i];
            float4 r;
            r.x = (v.x - mean) * inv_std * g.x + b.x;
            r.y = (v.y - mean) * inv_std * g.y + b.y;
            r.z = (v.z - mean) * inv_std * g.z + b.z;
            r.w = (v.w - mean) * inv_std * g.w + b.w;
            out4[i] = r;
        }
    } else {
        for (int i = tid; i < row_size; i += blockDim.x) {
            const float v = cached ? cache[i] : row_in[i];
            row_out[i] = (v - mean) * inv_std * gamma[i] + beta[i];
        }
    }
}
"""

_WARP_SIZE = 32
_MAX_BLOCK_SIZE = 256
_MAX_CACHED_ROW = 8192

_module = SourceModule(_KERNEL_SOURCE)
_kernel = _module.get_function("LayerNorm")
_kernel.prepare("PPPPifii" + "i")

_buffers = {}


def _device_buffer(name, nbytes):
    buf = _buffers.get(name)
    if buf is None or buf[1] < nbytes:
        buf = (cuda.mem_alloc(max(nbytes, 1)), nbytes)
        _buffers[name] = buf
    return buf[0]


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    x = np.ascontiguousarray(input, dtype=np.float32).ravel()
    g = np.ascontiguousarray(gamma, dtype=np.float32).ravel()
    b = np.ascontiguousarray(beta, dtype=np.float32).ravel()
    row_size = int(row_size)
    if x.size == 0:
        return np.empty(0, dtype=np.float32)
    row_count = x.size // row_size

    d_in = _device_buffer("in", x.nbytes)
    d_out = _device_buffer("out", x.nbytes)
    d_gamma = _device_buffer("gamma", g.nbytes)
    d_beta = _device_buffer("beta", b.nbytes)
    cuda.memcpy_htod(d_in, x)
    cuda.memcpy_htod(d_gamma, g)
    cuda.memcpy_htod(d_beta, b)

    vec4 = row_size % 4 == 0
    work = row_size // 4 if vec4 else row_size
    block_size = _WARP_SIZE
    while block_size < _MAX_BLOCK_SIZE and block_size < work:
        block_size *= 2

    cached = row_size <= _MAX_CACHED_ROW
    shared = row_size * 4 if cached else 0

    _kernel.prepared_call((row_count, 1), (block_size, 1, 1),
                          d_in, d_gamma, d_beta, d_out,
                          np.int32(row_size), np.float32(eps),
                          np.int32(cached), np.int32(vec4),
                          shared_size=shared)

    out = np.empty_like(x)
    cuda.memcpy_dtoh(out, d_out)
    return out