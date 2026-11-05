import operator

import numpy as np


_KERNEL_SOURCE = r"""
#include <math.h>

__global__ void layernorm_kernel(float* data, const float* gamma,
                                const float* beta, unsigned long long columns,
                                unsigned long long rows, float eps) {
    __shared__ float reduction[256];
    const unsigned int tid = threadIdx.x;
    for (unsigned long long row = blockIdx.x; row < rows; row += gridDim.x) {
        float* values = data + row * columns;
        // Center before summation to retain small differences at large offsets.
        const float anchor = values[0];
        float sum = 0.0f;
        for (unsigned long long col = tid; col < columns; col += blockDim.x) {
            sum += values[col] - anchor;
        }
        reduction[tid] = sum;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (tid < stride) reduction[tid] += reduction[tid + stride];
            __syncthreads();
        }
        const float mean_delta = reduction[0] / static_cast<float>(columns);
        __syncthreads();

        float squared_sum = 0.0f;
        for (unsigned long long col = tid; col < columns; col += blockDim.x) {
            const float delta = (values[col] - anchor) - mean_delta;
            squared_sum = fmaf(delta, delta, squared_sum);
        }
        reduction[tid] = squared_sum;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (tid < stride) reduction[tid] += reduction[tid + stride];
            __syncthreads();
        }
        const float inverse_std = rsqrtf(reduction[0] / static_cast<float>(columns) + eps);
        // All reads used for statistics finish before in-place output writes.
        __syncthreads();
        for (unsigned long long col = tid; col < columns; col += blockDim.x) {
            const float normalized = ((values[col] - anchor) - mean_delta) * inverse_std;
            values[col] = fmaf(gamma[col], normalized, beta[col]);
        }
        __syncthreads();
    }
}
"""


_compiled_kernel = None


def _get_kernel():
    global _compiled_kernel
    if _compiled_kernel is None:
        from pycuda.compiler import SourceModule
        from pycuda.tools import context_dependent_memoize

        @context_dependent_memoize
        def compile_kernel():
            module = SourceModule(_KERNEL_SOURCE)
            return module.get_function("layernorm_kernel")

        _compiled_kernel = compile_kernel
    return _compiled_kernel()


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    """Return flattened float32 row-wise LayerNorm using PyCUDA.

    input is a row-major flat matrix; gamma and beta contain row_size values.
    Normalization uses population variance and sqrt(variance + eps).
    """
    row_size = operator.index(row_size)
    if row_size <= 0:
        raise ValueError("row_size must be positive")
    eps = np.float32(eps)
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive in float32")

    values = np.ascontiguousarray(np.asarray(input, dtype=np.float32).reshape(-1))
    scale = np.ascontiguousarray(np.asarray(gamma, dtype=np.float32).reshape(-1))
    shift = np.ascontiguousarray(np.asarray(beta, dtype=np.float32).reshape(-1))
    if values.size % row_size:
        raise ValueError("input length must be divisible by row_size")
    if scale.size != row_size or shift.size != row_size:
        raise ValueError("gamma and beta must each contain row_size values")
    if values.size == 0:
        return np.empty(0, dtype=np.float32)

    import pycuda.driver as cuda

    # Respect a context provided by the caller, otherwise initialize one lazily.
    cuda.init()
    if cuda.Context.get_current() is None:
        import pycuda.autoinit  # noqa: F401
        if cuda.Context.get_current() is None:
            raise RuntimeError("No active PyCUDA context")
    kernel = _get_kernel()
    threads = min(256, max(32, 1 << (row_size - 1).bit_length()))
    rows = values.size // row_size
    allocations = []
    try:
        for host in (values, scale, shift):
            device = cuda.mem_alloc(host.nbytes)
            allocations.append(device)
            cuda.memcpy_htod(device, host)
        kernel(allocations[0], allocations[1], allocations[2],
               np.uint64(row_size), np.uint64(rows), eps,
               block=(threads, 1, 1), grid=(min(rows, 65535), 1, 1))
        output = np.empty(values.size, dtype=np.float32)
        # Synchronous copy waits for the kernel and surfaces execution failures.
        cuda.memcpy_dtoh(output, allocations[0])
        return output
    finally:
        for device in reversed(allocations):
            device.free()
