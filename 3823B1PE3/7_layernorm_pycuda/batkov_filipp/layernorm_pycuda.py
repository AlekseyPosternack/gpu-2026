import operator

import numpy as np


_CUDA_SOURCE = r"""
#include <math.h>

extern "C" __global__ void layernorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    const unsigned long long row_count,
    const unsigned long long row_size,
    const float eps)
{
    __shared__ double reduction[BLOCK_SIZE];
    __shared__ double shared_mean;
    __shared__ double shared_inverse_std;
    const unsigned int tid = threadIdx.x;

#if ITEMS_PER_THREAD > 0
    float values[ITEMS_PER_THREAD];
#endif

    for (unsigned long long row = blockIdx.x;
         row < row_count; row += gridDim.x) {
        const unsigned long long base = row * row_size;
        double local_sum = 0.0;

#if ITEMS_PER_THREAD > 0
        #pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
            const unsigned long long column =
                tid + static_cast<unsigned long long>(item) * BLOCK_SIZE;
            values[item] = column < row_size ? input[base + column] : 0.0f;
            local_sum += static_cast<double>(values[item]);
        }
#else
        for (unsigned long long column = tid;
             column < row_size; column += BLOCK_SIZE) {
            local_sum += static_cast<double>(input[base + column]);
        }
#endif

        reduction[tid] = local_sum;
        __syncthreads();
        #pragma unroll
        for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
            if (tid < static_cast<unsigned int>(stride)) {
                reduction[tid] += reduction[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            shared_mean = reduction[0] / static_cast<double>(row_size);
        }
        __syncthreads();
        const double mean = shared_mean;

        double local_variance = 0.0;
#if ITEMS_PER_THREAD > 0
        #pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
            const unsigned long long column =
                tid + static_cast<unsigned long long>(item) * BLOCK_SIZE;
            // Padding does not contribute to the variance.
            if (column < row_size) {
                const double difference = static_cast<double>(values[item]) - mean;
                local_variance += difference * difference;
            }
        }
#else
        for (unsigned long long column = tid;
             column < row_size; column += BLOCK_SIZE) {
            const double difference =
                static_cast<double>(input[base + column]) - mean;
            local_variance += difference * difference;
        }
#endif

        reduction[tid] = local_variance;
        __syncthreads();
        #pragma unroll
        for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
            if (tid < static_cast<unsigned int>(stride)) {
                reduction[tid] += reduction[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            const double variance = reduction[0] / static_cast<double>(row_size);
            shared_inverse_std = 1.0 / sqrt(variance + static_cast<double>(eps));
        }
        __syncthreads();
        const double inverse_std = shared_inverse_std;

#if ITEMS_PER_THREAD > 0
        #pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
            const unsigned long long column =
                tid + static_cast<unsigned long long>(item) * BLOCK_SIZE;
            if (column < row_size) {
                const float normalized = static_cast<float>(
                    (static_cast<double>(values[item]) - mean) * inverse_std);
                output[base + column] = normalized * gamma[column] + beta[column];
            }
        }
#else
        for (unsigned long long column = tid;
             column < row_size; column += BLOCK_SIZE) {
            const float normalized = static_cast<float>(
                (static_cast<double>(input[base + column]) - mean) * inverse_std);
            output[base + column] = normalized * gamma[column] + beta[column];
        }
#endif
        // All readers finish before shared statistics are reused for another row.
        __syncthreads();
    }
}
"""


_kernel_cache = None


def _get_kernel(block_size, items_per_thread):
    global _kernel_cache

    if _kernel_cache is None:
        from pycuda.compiler import SourceModule
        from pycuda.tools import context_dependent_memoize

        @context_dependent_memoize
        def compile_kernel(block_size, items_per_thread):
            source = (
                f"#define BLOCK_SIZE {block_size}\n"
                f"#define ITEMS_PER_THREAD {items_per_thread}\n"
                + _CUDA_SOURCE
            )
            module = SourceModule(source, options=["-std=c++17"], no_extern_c=True)
            kernel = module.get_function("layernorm_kernel")
            kernel.prepare("PPPPQQf")
            return module, kernel

        _kernel_cache = compile_kernel

    return _kernel_cache(block_size, items_per_thread)


def _float32_vector(values, name):
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a flattened one-dimensional array")
    return np.ascontiguousarray(array)


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    """Apply LayerNorm to rows of a flattened matrix and return flat float32.

    ``gamma`` and ``beta`` must contain ``row_size`` elements. Variance uses
    division by ``row_size``; ``eps`` must be finite and positive in float32.
    CUDA kernels are compiled lazily and cached per CUDA context and shape class.
    """
    row_size = operator.index(row_size)
    if row_size <= 0 or row_size & (row_size - 1):
        raise ValueError("row_size must be a positive power of two")

    values = _float32_vector(input, "input")
    scales = _float32_vector(gamma, "gamma")
    shifts = _float32_vector(beta, "beta")
    if values.size % row_size != 0:
        raise ValueError("Input length must be divisible by row_size")
    if scales.size != row_size or shifts.size != row_size:
        raise ValueError("gamma and beta must contain row_size elements")
    if np.ndim(eps) != 0:
        raise ValueError("eps must be a scalar")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        epsilon = np.float32(eps)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("eps must be finite and positive in float32")
    if values.size == 0:
        return np.empty(0, dtype=np.float32)

    import pycuda.driver as cuda

    cuda.init()
    if cuda.Context.get_current() is None:
        import pycuda.autoinit  # noqa: F401

    if cuda.Context.get_current() is None:
        raise RuntimeError("A current PyCUDA context is required in this thread")

    row_count = values.size // row_size
    block_size = min(256, max(32, row_size))
    items_per_thread = max(1, row_size // block_size)
    if items_per_thread > 64:
        items_per_thread = 0
    module, kernel = _get_kernel(block_size, items_per_thread)

    allocations = []
    pending_work = False
    try:
        device_input = cuda.mem_alloc(values.nbytes)
        allocations.append(device_input)
        device_gamma = cuda.mem_alloc(scales.nbytes)
        allocations.append(device_gamma)
        device_beta = cuda.mem_alloc(shifts.nbytes)
        allocations.append(device_beta)
        device_output = cuda.mem_alloc(values.nbytes)
        allocations.append(device_output)

        cuda.memcpy_htod(device_input, values)
        cuda.memcpy_htod(device_gamma, scales)
        cuda.memcpy_htod(device_beta, shifts)

        pending_work = True
        kernel.prepared_call(
            (min(row_count, 65535), 1, 1),
            (block_size, 1, 1),
            device_input, device_gamma, device_beta, device_output,
            np.uint64(row_count), np.uint64(row_size), epsilon,
        )
        
        result = np.empty_like(values)
        cuda.memcpy_dtoh(result, device_output)
        pending_work = False
        return result
    finally:
        if pending_work:
            try:
                cuda.Context.synchronize()
            except cuda.Error:
                pass
        for allocation in reversed(allocations):
            try:
                allocation.free()
            except cuda.Error:
                pass
