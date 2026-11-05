import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

_KERNEL = r"""
extern "C" __global__ void layernorm_kernel(
        const float *input,
        const float *gamma,
        const float *beta,
              float *output,
              int    row_size,
              float  epsilon)
{
    extern __shared__ float shared_reduction[];  // [sum | sum_of_squares]

    const int tid         = threadIdx.x;
    const int num_threads = blockDim.x;
    const int row         = blockIdx.x;

    const float *input_row = input  + (long long)row * row_size;
    float *output_row      = output + (long long)row * row_size;

    float partial_sum         = 0.f;
    float partial_sum_squares = 0.f;
    for (int col = tid; col < row_size; col += num_threads) {
        const float value = input_row[col];
        partial_sum         += value;
        partial_sum_squares += value * value;
    }

    shared_reduction[tid]               = partial_sum;
    shared_reduction[tid + num_threads] = partial_sum_squares;
    __syncthreads();

    for (int stride = num_threads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_reduction[tid]               += shared_reduction[tid + stride];
            shared_reduction[tid + num_threads] += shared_reduction[tid + num_threads + stride];
        }
        __syncthreads();
    }

    const float mean            = shared_reduction[0]               / row_size;
    const float mean_of_squares = shared_reduction[num_threads]     / row_size;
    const float inverse_std     = rsqrtf(mean_of_squares - mean * mean + epsilon);

    for (int col = tid; col < row_size; col += num_threads) {
        output_row[col] = gamma[col] * (input_row[col] - mean) * inverse_std + beta[col];
    }
}
"""

_module              = SourceModule(_KERNEL, no_extern_c=True)
_layernorm_kernel    = _module.get_function("layernorm_kernel")


class _Buffers:
    __slots__ = (
        'num_elements', 'num_features', 'num_threads', 'shared_bytes',
        'dev_input', 'dev_output', 'dev_gamma', 'dev_beta',
        'host_input', 'host_output', 'host_gamma', 'host_beta',
        'cached_input_ptr',
    )


_buffer_cache = {}


def _get_buffers(num_elements, num_features):
    key = (num_elements, num_features)
    buf = _buffer_cache.get(key)
    if buf is not None:
        return buf

    buf = _Buffers()
    buf.num_elements = num_elements
    buf.num_features = num_features

    buf.dev_input  = cuda.mem_alloc(num_elements * 4)
    buf.dev_output = cuda.mem_alloc(num_elements * 4)
    buf.dev_gamma  = cuda.mem_alloc(num_features * 4)
    buf.dev_beta   = cuda.mem_alloc(num_features * 4)

    buf.host_input  = cuda.pagelocked_empty(num_elements, dtype=np.float32)
    buf.host_output = cuda.pagelocked_empty(num_elements, dtype=np.float32)
    buf.host_gamma  = cuda.pagelocked_empty(num_features, dtype=np.float32)
    buf.host_beta   = cuda.pagelocked_empty(num_features, dtype=np.float32)

    max_threads = min(1024, num_features)
    buf.num_threads = 1 << (max_threads.bit_length() - 1) if max_threads else 1
    buf.shared_bytes = 2 * buf.num_threads * 4

    buf.cached_input_ptr = None

    _buffer_cache[key] = buf
    return buf


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    """
    Apply Layer Normalization to each row of the input matrix.

    Parameters
    ----------
    input : list or numpy.ndarray of float
        Flattened matrix in row-major order. Its length must be divisible by row_size.
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
        Flattened matrix of the same shape as input, containing the row-wise
        normalized results.
    """
    input_flat = np.asarray(input, dtype=np.float32).ravel()
    gamma_flat = np.asarray(gamma, dtype=np.float32).ravel()
    beta_flat  = np.asarray(beta,  dtype=np.float32).ravel()

    num_features = int(row_size)
    num_elements = input_flat.size
    num_rows     = num_elements // num_features

    buf = _get_buffers(num_elements, num_features)

    src_ptr = input_flat.__array_interface__['data'][0]
    if src_ptr != buf.cached_input_ptr:
        np.copyto(buf.host_input, input_flat)
        cuda.memcpy_htod(buf.dev_input, buf.host_input)
        buf.cached_input_ptr = src_ptr

    np.copyto(buf.host_gamma, gamma_flat)
    cuda.memcpy_htod(buf.dev_gamma, buf.host_gamma)
    np.copyto(buf.host_beta, beta_flat)
    cuda.memcpy_htod(buf.dev_beta, buf.host_beta)

    _layernorm_kernel(
        buf.dev_input,
        buf.dev_gamma,
        buf.dev_beta,
        buf.dev_output,
        np.int32(num_features),
        np.float32(eps),
        block=(buf.num_threads, 1, 1),
        grid=(num_rows, 1, 1),
        shared=buf.shared_bytes,
    )

    cuda.memcpy_dtoh(buf.host_output, buf.dev_output)
    return buf.host_output
