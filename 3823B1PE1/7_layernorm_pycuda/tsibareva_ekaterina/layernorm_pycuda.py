import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

_kernel_module = None
_kernel = None


def _get_kernel():
    global _kernel_module, _kernel
    if _kernel is None:
        _kernel_module = SourceModule(r"""
extern "C" __global__ void layernorm_kernel(
    const float *x,
    const float *gamma,
    const float *beta,
    float *y,
    int row_size,
    int rows,
    float eps
)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    extern __shared__ float sdata[];

    float sum = 0.0f;
    for (int i = tid; i < row_size; i += nthreads) {
        sum += x[row * row_size + i];
    }

    sdata[tid] = sum;
    __syncthreads();

    for (int stride = nthreads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    float mean = sdata[0] / (float)row_size;
    __syncthreads();

    float var_sum = 0.0f;
    for (int i = tid; i < row_size; i += nthreads) {
        float d = x[row * row_size + i] - mean;
        var_sum += d * d;
    }

    sdata[tid] = var_sum;
    __syncthreads();

    for (int stride = nthreads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    float inv_std = rsqrtf(sdata[0] / (float)row_size + eps);

    for (int i = tid; i < row_size; i += nthreads) {
        int idx = row * row_size + i;
        float norm = (x[idx] - mean) * inv_std;
        y[idx] = gamma[i] * norm + beta[i];
    }
}
""")
        _kernel = _kernel_module.get_function("layernorm_kernel")
    return _kernel


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
    # TODO: Implement using PyCUDA
    x = np.ascontiguousarray(input, dtype=np.float32).ravel()
    g = np.ascontiguousarray(gamma, dtype=np.float32).ravel()
    b = np.ascontiguousarray(beta, dtype=np.float32).ravel()

    if row_size <= 0 or x.size % row_size != 0:
        raise ValueError("input length must be divisible by row_size")

    rows = x.size // row_size
    out = np.empty_like(x)

    if rows == 0:
        return out

    x_d = cuda.mem_alloc(x.nbytes)
    g_d = cuda.mem_alloc(g.nbytes)
    b_d = cuda.mem_alloc(b.nbytes)
    out_d = cuda.mem_alloc(out.nbytes)

    cuda.memcpy_htod(x_d, x)
    cuda.memcpy_htod(g_d, g)
    cuda.memcpy_htod(b_d, b)

    block_size = min(row_size, 1024)
    kernel = _get_kernel()

    kernel(
        x_d,
        g_d,
        b_d,
        out_d,
        np.int32(row_size),
        np.int32(rows),
        np.float32(eps),
        block=(block_size, 1, 1),
        grid=(rows, 1, 1),
        shared=block_size * np.dtype(np.float32).itemsize
    )

    cuda.memcpy_dtoh(out, out_d)
    return out