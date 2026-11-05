import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda.tools import DeviceMemoryPool, context_dependent_memoize
_SOURCE = r"""
template<int Stage>
__device__ float reduce_sum(float value) {
    __shared__ float partial[8];
    const int lane = threadIdx.x % 32;
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    if (lane == 0) {
        partial[threadIdx.x / 32] = value;
    }
    __syncthreads();
    value = threadIdx.x < 8 ? partial[threadIdx.x] : 0.0f;
    if (threadIdx.x < 32) {
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
        if (lane == 0) {
            partial[0] = value;
        }
    }
    __syncthreads();
    return partial[0];
}
extern "C" __global__ void layernorm(float* data, const float* gamma, const float* beta, int width, float eps) {
    float* row = data + static_cast<size_t>(blockIdx.x) * width;
    const float anchor = row[0];
    const bool cached = width <= 16384;
    float values[64];
    float sum = 0.0f;
    if (cached) {
#pragma unroll
        for (int i = 0; i < 64; ++i) {
            const int col = threadIdx.x + i * 256;
            values[i] = col < width ? row[col] - anchor : 0.0f;
            sum += values[i];
        }
    } else {
        for (int col = threadIdx.x; col < width; col += 256) {
            sum += row[col] - anchor;
        }
    }
    const float mean = reduce_sum<0>(sum) / width;
    float variance = 0.0f;
    if (cached) {
#pragma unroll
        for (int i = 0; i < 64; ++i) {
            const int col = threadIdx.x + i * 256;
            values[i] = col < width ? values[i] - mean : 0.0f;
            variance = fmaf(values[i], values[i], variance);
        }
    } else {
        for (int col = threadIdx.x; col < width; col += 256) {
            const float value = (row[col] - anchor) - mean;
            variance = fmaf(value, value, variance);
        }
    }
    const float inverse = rsqrtf(reduce_sum<1>(variance) / width + eps);
    if (cached) {
#pragma unroll
        for (int i = 0; i < 64; ++i) {
            const int col = threadIdx.x + i * 256;
            if (col < width) {
                row[col] = values[i] * inverse * gamma[col] + beta[col];
            }
        }
    } else {
        for (int col = threadIdx.x; col < width; col += 256) {
            row[col] = ((row[col] - anchor) - mean) * inverse * gamma[col] + beta[col];
        }
    }
}
"""
@context_dependent_memoize
def _resources():
    module = SourceModule(_SOURCE, options=["-O3"], no_extern_c=True)
    kernel = module.get_function("layernorm")
    kernel.prepare("PPPif")
    return kernel, DeviceMemoryPool()
def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    if row_size <= 0 or int(row_size) != row_size:
        raise ValueError("row_size must be a positive integer")
    row_size = int(row_size)
    eps = np.float32(eps)
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    x = np.ascontiguousarray(input, dtype=np.float32).reshape(-1)
    g = np.ascontiguousarray(gamma, dtype=np.float32).reshape(-1)
    b = np.ascontiguousarray(beta, dtype=np.float32).reshape(-1)
    if x.size % row_size or g.size != row_size or b.size != row_size:
        raise ValueError("Input, gamma or beta size does not match row_size")
    output = np.empty_like(x)
    if x.size == 0:
        return output
    kernel, pool = _resources()
    allocations = [pool.allocate(array.nbytes) for array in (x, g, b)]
    try:
        for device, array in zip(allocations, (x, g, b)):
            cuda.memcpy_htod(device, array)
        kernel.prepared_call((x.size // row_size, 1, 1), (256, 1, 1),
                             *allocations, np.int32(row_size), eps)
        cuda.memcpy_dtoh(output, allocations[0])
    finally:
        for device in allocations:
            device.free()
    return output
