import operator
import threading

import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import DEFAULT_NVCC_FLAGS, SourceModule
from pycuda.tools import PageLockedMemoryPool, context_dependent_memoize


_SOURCE = r"""
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

template<int Vectors>
__device__ __forceinline__ void normalize_cached(
    float* data, const float* gamma, const float* beta, int width, float eps) {
    using Reduction = cub::BlockReduce<float, 256>;
    __shared__ typename Reduction::TempStorage temporary;
    __shared__ float statistic;
    float* values = data + static_cast<size_t>(blockIdx.x) * width;
    float4* row = reinterpret_cast<float4*>(values);
    const float anchor = values[0];
    const int count = width / 4;
    float4 cached[Vectors];
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < Vectors; ++i) {
        const int index = threadIdx.x + i * 256;
        cached[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (index < count) {
            cached[i] = row[index];
            cached[i].x -= anchor;
            cached[i].y -= anchor;
            cached[i].z -= anchor;
            cached[i].w -= anchor;
        }
        sum += (cached[i].x + cached[i].y) + (cached[i].z + cached[i].w);
    }
    sum = Reduction(temporary).Sum(sum);
    if (threadIdx.x == 0) statistic = sum / width;
    __syncthreads();
    const float mean = statistic;
    float squares = 0.0f;
#pragma unroll
    for (int i = 0; i < Vectors; ++i) {
        if (threadIdx.x + i * 256 < count) {
            cached[i].x -= mean;
            cached[i].y -= mean;
            cached[i].z -= mean;
            cached[i].w -= mean;
            squares += (cached[i].x * cached[i].x + cached[i].y * cached[i].y) +
                       (cached[i].z * cached[i].z + cached[i].w * cached[i].w);
        }
    }
    squares = Reduction(temporary).Sum(squares);
    if (threadIdx.x == 0) statistic = rsqrtf(squares / width + eps);
    __syncthreads();
    const float scale = statistic;
#pragma unroll
    for (int i = 0; i < Vectors; ++i) {
        const int index = threadIdx.x + i * 256;
        if (index < count) {
            const float4 g = reinterpret_cast<const float4*>(gamma)[index];
            const float4 b = reinterpret_cast<const float4*>(beta)[index];
            row[index] = make_float4(
                fmaf(cached[i].x * scale, g.x, b.x),
                fmaf(cached[i].y * scale, g.y, b.y),
                fmaf(cached[i].z * scale, g.z, b.z),
                fmaf(cached[i].w * scale, g.w, b.w));
        }
    }
}

extern "C" __global__ void normalize_generic(
    float* data, const float* gamma, const float* beta, int width, float eps) {
    using Reduction = cub::BlockReduce<double, 256>;
    __shared__ typename Reduction::TempStorage temporary;
    __shared__ double statistic;
    float* row = data + static_cast<size_t>(blockIdx.x) * width;
    const double anchor = row[0];
    double sum = 0.0;
    for (int column = threadIdx.x; column < width; column += 256) {
        sum += static_cast<double>(row[column]) - anchor;
    }
    sum = Reduction(temporary).Sum(sum);
    if (threadIdx.x == 0) statistic = sum / width;
    __syncthreads();
    const double mean = statistic;
    double squares = 0.0;
    for (int column = threadIdx.x; column < width; column += 256) {
        const double difference = (static_cast<double>(row[column]) - anchor) - mean;
        squares += difference * difference;
    }
    squares = Reduction(temporary).Sum(squares);
    if (threadIdx.x == 0) statistic = 1.0 / sqrt(squares / width + eps);
    __syncthreads();
    const double scale = statistic;
    for (int column = threadIdx.x; column < width; column += 256) {
        const double difference = (static_cast<double>(row[column]) - anchor) - mean;
        row[column] = static_cast<float>(difference * scale * gamma[column] + beta[column]);
    }
}
"""

for _vectors in (1, 2, 4, 8, 16):
    _SOURCE += f"""
extern "C" __global__ void __launch_bounds__(256) normalize_{_vectors}(
    float* data, const float* gamma, const float* beta, int width, float eps) {{
    normalize_cached<{_vectors}>(data, gamma, beta, width, eps);
}}
"""


class _Workspace:
    def __init__(self):
        options = [*DEFAULT_NVCC_FLAGS, "-O3", "-std=c++17"]
        self.module = SourceModule(_SOURCE, options=options, no_extern_c=True)
        self.kernels = {}
        for count in (1, 2, 4, 8, 16):
            kernel = self.module.get_function(f"normalize_{count}")
            kernel.prepare("PPPif")
            self.kernels[count] = kernel
        self.generic = self.module.get_function("normalize_generic")
        self.generic.prepare("PPPif")
        self.data = None
        self.gamma = None
        self.beta = None
        self.capacity = 0
        self.features = 0
        self.outputs = PageLockedMemoryPool()

    def reserve(self, count, width):
        if count > self.capacity:
            self.data = cuda.mem_alloc(count * 4)
            self.capacity = count
        if width > self.features:
            self.gamma = cuda.mem_alloc(width * 4)
            self.beta = cuda.mem_alloc(width * 4)
            self.features = width

    def select(self, width):
        if width % 4 != 0 or width > 16384:
            return self.generic
        for count in (1, 2, 4, 8, 16):
            if width <= count * 1024:
                return self.kernels[count]
        return self.generic


@context_dependent_memoize
def _workspace():
    return _Workspace()


_lock = threading.RLock()


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    width = operator.index(row_size)
    if width <= 0 or width > np.iinfo(np.int32).max:
        raise ValueError("row_size must be a positive 32-bit integer")
    epsilon = np.float32(eps)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("eps must be positive and finite")
    values = np.ascontiguousarray(input, dtype=np.float32).reshape(-1)
    scale = np.ascontiguousarray(gamma, dtype=np.float32).reshape(-1)
    bias = np.ascontiguousarray(beta, dtype=np.float32).reshape(-1)
    if values.size % width != 0:
        raise ValueError("Input size must be divisible by row_size")
    if scale.size != width or bias.size != width:
        raise ValueError("gamma and beta must contain row_size elements")
    if values.size == 0:
        return np.empty(0, dtype=np.float32)
    rows = values.size // width
    if rows > np.iinfo(np.int32).max:
        raise ValueError("Too many rows for a CUDA grid")

    with _lock:
        cuda.init()
        if cuda.Context.get_current() is None:
            import pycuda.autoinit
        state = _workspace()
        state.reserve(values.size, width)
        output = state.outputs.allocate((values.size,), np.float32)
        cuda.memcpy_htod(state.data, values)
        cuda.memcpy_htod(state.gamma, scale)
        cuda.memcpy_htod(state.beta, bias)
        state.select(width).prepared_call(
            (rows, 1, 1), (256, 1, 1),
            state.data, state.gamma, state.beta, np.int32(width), epsilon
        )
        cuda.memcpy_dtoh(output, state.data)
    return output
