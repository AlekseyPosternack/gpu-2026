
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda

from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from pycuda.compiler import SourceModule
from pycuda.tools import DeviceMemoryPool


GPU_POOL = DeviceMemoryPool()
CPU_POOL = ThreadPoolExecutor(max_workers=1)


CUDA_SOURCE = r"""
#include <math.h>

__device__ __forceinline__
float warp_reduce_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(
            0xffffffff, value, offset
        );
    }
    return value;
}

extern "C" __global__
void layernorm_kernel(
    float* data,
    const float* gamma,
    const float* beta,
    int row_size,
    float eps
) {
    __shared__ float warp_sums[8];
    __shared__ float shared_mean;
    __shared__ float shared_inv_std;

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;

    size_t base = (size_t)blockIdx.x * row_size;

    // Calculate mean
    float local_sum = 0.0f;

    for (int i = tid; i < row_size; i += 256) {
        local_sum += data[base + i];
    }

    local_sum = warp_reduce_sum(local_sum);

    if (lane == 0) {
        warp_sums[warp_id] = local_sum;
    }

    __syncthreads();

    if (warp_id == 0) {
        float total = lane < 8 ? warp_sums[lane] : 0.0f;
        total = warp_reduce_sum(total);

        if (lane == 0) {
            shared_mean = total / row_size;
        }
    }

    __syncthreads();

    float mean = shared_mean;

    // Calculate variance
    float local_var = 0.0f;

    for (int i = tid; i < row_size; i += 256) {
        float diff = data[base + i] - mean;
        local_var += diff * diff;
    }

    local_var = warp_reduce_sum(local_var);

    if (lane == 0) {
        warp_sums[warp_id] = local_var;
    }

    __syncthreads();

    if (warp_id == 0) {
        float total = lane < 8 ? warp_sums[lane] : 0.0f;
        total = warp_reduce_sum(total);

        if (lane == 0) {
            float variance = total / row_size;
            shared_inv_std = rsqrtf(variance + eps);
        }
    }

    __syncthreads();

    float inv_std = shared_inv_std;

    // Normalize in place
    for (int i = tid; i < row_size; i += 256) {
        float value = data[base + i];

        data[base + i] =
            (value - mean) * inv_std * gamma[i] + beta[i];
    }
}
"""


@lru_cache(maxsize=1)
def _get_kernel():
    module = SourceModule(
        CUDA_SOURCE,
        options=["-O3"]
    )
    return module.get_function("layernorm_kernel")


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    if row_size <= 0:
        raise ValueError("row_size must be positive")

    if eps <= 0:
        raise ValueError("eps must be positive")

    x = np.ascontiguousarray(
        np.asarray(input, dtype=np.float32).reshape(-1)
    )

    g = np.ascontiguousarray(
        np.asarray(gamma, dtype=np.float32).reshape(-1)
    )

    b = np.ascontiguousarray(
        np.asarray(beta, dtype=np.float32).reshape(-1)
    )

    if x.size % row_size != 0:
        raise ValueError("Invalid input size")

    if g.size != row_size or b.size != row_size:
        raise ValueError("Invalid gamma or beta size")

    output = np.empty_like(x)

    if x.size == 0:
        return output

    row_count = x.size // row_size
    allocations = []

    # Touch large output buffers in a separate CPU thread.
    # Small arrays do not need this extra operation.
    touch_future = None

    if output.nbytes >= 32 * 1024 * 1024:
        touch_future = CPU_POOL.submit(output.fill, 0.0)

    try:
        # Allocate or reuse GPU buffers
        d_data = GPU_POOL.allocate(x.nbytes)
        allocations.append(d_data)

        d_gamma = GPU_POOL.allocate(g.nbytes)
        allocations.append(d_gamma)

        d_beta = GPU_POOL.allocate(b.nbytes)
        allocations.append(d_beta)

        # Transfer data while CPU prepares output memory
        cuda.memcpy_htod(d_data, x)
        cuda.memcpy_htod(d_gamma, g)
        cuda.memcpy_htod(d_beta, b)

        kernel = _get_kernel()

        kernel(
            d_data,
            d_gamma,
            d_beta,
            np.int32(row_size),
            np.float32(eps),
            block=(256, 1, 1),
            grid=(int(row_count), 1, 1)
        )

        # Ensure CPU initialization finishes before GPU writes
        if touch_future is not None:
            touch_future.result()

        # Copy result to CPU
        cuda.memcpy_dtoh(output, d_data)

    finally:
        if touch_future is not None:
            touch_future.result()

        for allocation in reversed(allocations):
            allocation.free()

    return output
