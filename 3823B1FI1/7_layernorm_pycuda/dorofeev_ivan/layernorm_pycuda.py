import ctypes
import sys

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

THREADS = 256
MAX_VECS = 16
CHUNK = 1 << 22
SMALL = 1 << 23
PAGE = 4096
STREAMS = 4

KERNELS = r"""
#define THREADS 256
#define WARPS (THREADS / 32)

__device__ float block_sum(float value, float* shared) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    if (threadIdx.x % 32 == 0) {
        shared[threadIdx.x / 32] = value;
    }
    __syncthreads();
    float result = 0.0f;
    for (int i = 0; i < WARPS; ++i) {
        result += shared[i];
    }
    __syncthreads();
    return result;
}

template <int VECS>
__device__ void layernorm_cached(float* data, const float* gamma, const float* beta, int row_size, float eps) {
    __shared__ float shared[WARPS];
    float4* row = reinterpret_cast<float4*>(data + (size_t)blockIdx.x * row_size);
    const float4* g = reinterpret_cast<const float4*>(gamma);
    const float4* b = reinterpret_cast<const float4*>(beta);
    const int vecs = row_size / 4;
    const float shift = data[(size_t)blockIdx.x * row_size];

    float4 v[VECS];
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < VECS; ++i) {
        const int j = threadIdx.x + i * THREADS;
        if (j < vecs) {
            v[i] = row[j];
            v[i].x -= shift;
            v[i].y -= shift;
            v[i].z -= shift;
            v[i].w -= shift;
        } else {
            v[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        sum += (v[i].x + v[i].y) + (v[i].z + v[i].w);
    }
    const float mean = block_sum(sum, shared) / row_size;

    float squares = 0.0f;
#pragma unroll
    for (int i = 0; i < VECS; ++i) {
        if (threadIdx.x + i * THREADS < vecs) {
            v[i].x -= mean;
            v[i].y -= mean;
            v[i].z -= mean;
            v[i].w -= mean;
            squares += (v[i].x * v[i].x + v[i].y * v[i].y) + (v[i].z * v[i].z + v[i].w * v[i].w);
        }
    }
    const float inv = rsqrtf(block_sum(squares, shared) / row_size + eps);

#pragma unroll
    for (int i = 0; i < VECS; ++i) {
        const int j = threadIdx.x + i * THREADS;
        if (j < vecs) {
            const float4 gj = g[j];
            const float4 bj = b[j];
            row[j] = make_float4(v[i].x * inv * gj.x + bj.x, v[i].y * inv * gj.y + bj.y,
                                 v[i].z * inv * gj.z + bj.z, v[i].w * inv * gj.w + bj.w);
        }
    }
}

#define CACHED(VECS)                                                                                       \
    extern "C" __global__ void __launch_bounds__(THREADS)                                                  \
        layernorm_##VECS(float* data, const float* gamma, const float* beta, int row_size, float eps) {    \
        layernorm_cached<VECS>(data, gamma, beta, row_size, eps);                                          \
    }

CACHED(1)
CACHED(2)
CACHED(4)
CACHED(8)
CACHED(16)

extern "C" __global__ void __launch_bounds__(THREADS)
    layernorm_generic(float* data, const float* gamma, const float* beta, int row_size, float eps) {
    __shared__ float shared[WARPS];
    float* row = data + (size_t)blockIdx.x * row_size;
    const float shift = row[0];

    float sum = 0.0f;
    for (int j = threadIdx.x; j < row_size; j += THREADS) {
        sum += row[j] - shift;
    }
    const float mean = block_sum(sum, shared) / row_size;

    float squares = 0.0f;
    for (int j = threadIdx.x; j < row_size; j += THREADS) {
        const float d = (row[j] - shift) - mean;
        squares += d * d;
    }
    const float inv = rsqrtf(block_sum(squares, shared) / row_size + eps);

    for (int j = threadIdx.x; j < row_size; j += THREADS) {
        row[j] = ((row[j] - shift) - mean) * inv * gamma[j] + beta[j];
    }
}
"""

_module = SourceModule(KERNELS, no_extern_c=True)
_cached = {vecs: _module.get_function("layernorm_%d" % vecs) for vecs in (1, 2, 4, 8, 16)}
_generic = _module.get_function("layernorm_generic")
for _function in list(_cached.values()) + [_generic]:
    _function.prepare("PPPif")


class _Context:
    def __init__(self):
        self.capacity = 0
        self.row_capacity = 0
        self.data = None
        self.gamma = None
        self.beta = None
        self.streams = [cuda.Stream() for _ in range(STREAMS)]
        self.out_streams = [cuda.Stream() for _ in range(STREAMS)]

    def reserve(self, n, row_size):
        if n > self.capacity:
            self.data = cuda.mem_alloc(n * 4)
            self.capacity = n
        if row_size > self.row_capacity:
            self.gamma = cuda.mem_alloc(row_size * 4)
            self.beta = cuda.mem_alloc(row_size * 4)
            self.row_capacity = row_size


_context = None


def _load_driver():
    try:
        driver = ctypes.CDLL("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")
        register = driver.cuMemHostRegister_v2
        unregister = driver.cuMemHostUnregister
    except (OSError, AttributeError):
        return None, None
    register.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
    register.restype = ctypes.c_int
    unregister.argtypes = [ctypes.c_void_p]
    unregister.restype = ctypes.c_int
    return register, unregister


_register, _unregister = _load_driver()


def _pin(array, pinned):
    if _register is not None and _register(array.ctypes.data, array.nbytes, 0) == 0:
        pinned.append(array)
    return array


def _kernel(row_size):
    vecs = row_size // 4
    if row_size % 4 != 0 or vecs > MAX_VECS * THREADS:
        return _generic
    for count in (1, 2, 4, 8, 16):
        if vecs <= count * THREADS:
            return _cached[count]


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
    global _context
    x = np.ascontiguousarray(input, dtype=np.float32).reshape(-1)
    g = np.ascontiguousarray(gamma, dtype=np.float32).reshape(-1)
    b = np.ascontiguousarray(beta, dtype=np.float32).reshape(-1)
    n = x.size
    row_size = int(row_size)
    out = np.empty(n, dtype=np.float32)
    if n == 0:
        return out

    if _context is None:
        _context = _Context()
    ctx = _context
    ctx.reserve(n, row_size)
    cuda.memcpy_htod(ctx.gamma, g)
    cuda.memcpy_htod(ctx.beta, b)
    kernel = _kernel(row_size)
    eps = np.float32(eps)
    base = int(ctx.data)

    def launch(offset, rows, stream):
        kernel.prepared_async_call((rows, 1), (THREADS, 1, 1), stream, base + offset * 4, ctx.gamma, ctx.beta,
                                   row_size, eps)

    if n < SMALL:
        cuda.memcpy_htod(ctx.data, x)
        launch(0, n // row_size, None)
        cuda.memcpy_dtoh(out, ctx.data)
        return out

    part = max(1, CHUNK // row_size) * row_size
    parts = (n + part - 1) // part
    shift = out.ctypes.data % PAGE // 4
    chunks = (n + shift + part - 1) // part

    def begin(c):
        return 0 if c == 0 else c * part - shift

    def end(c):
        return min(n, (c + 1) * part - shift)

    pinned = []
    _pin(x, pinned)
    computed = []
    for p in range(parts):
        stream = ctx.streams[p % STREAMS]
        lo, hi = p * part, min(n, (p + 1) * part)
        cuda.memcpy_htod_async(base + lo * 4, x[lo:hi], stream)
        launch(lo, (hi - lo) // row_size, stream)
        event = cuda.Event(cuda.event_flags.DISABLE_TIMING)
        event.record(stream)
        computed.append(event)

    for c in range(chunks):
        lo, hi = begin(c), end(c)
        stream = ctx.out_streams[c % STREAMS]
        chunk = _pin(out[lo:hi], pinned)
        for p in range(lo // part, (hi - 1) // part + 1):
            stream.wait_for_event(computed[p])
        cuda.memcpy_dtoh_async(chunk, base + lo * 4, stream)

    for stream in ctx.out_streams:
        stream.synchronize()
    for array in pinned:
        _unregister(array.ctypes.data)
    return out
