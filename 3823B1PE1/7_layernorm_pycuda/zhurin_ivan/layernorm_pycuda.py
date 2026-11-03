import numpy as np
import pycuda.autoinit  # noqa: F401  (creates the CUDA context)
import pycuda.driver as drv
from pycuda.compiler import SourceModule

_BLOCK = 256  # power of two, required by the tree reduction in the kernel

_KERNEL_SRC = r"""
extern "C" __global__ void layernorm_kernel(const float* __restrict__ input,
                                            const float* __restrict__ gamma,
                                            const float* __restrict__ beta,
                                            float* __restrict__ output,
                                            int row_size, float eps) {
    extern __shared__ float sdata[];

    const int row = blockIdx.x;
    const float* in = input + (size_t)row * row_size;
    float* out = output + (size_t)row * row_size;

    // 1. mean
    float local = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        local += in[i];
    }
    sdata[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float mean = sdata[0] / row_size;
    __syncthreads();

    // 2. variance (two-pass: more accurate than E[x^2] - mean^2)
    float local_var = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        float d = in[i] - mean;
        local_var += d * d;
    }
    sdata[threadIdx.x] = local_var;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv_std = rsqrtf(sdata[0] / row_size + eps);

    // 3. normalize, scale, shift
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        out[i] = gamma[i] * (in[i] - mean) * inv_std + beta[i];
    }
}
"""

_kernel = None


def _get_kernel():
    # Compile once and reuse on every call.
    global _kernel
    if _kernel is None:
        _kernel = SourceModule(_KERNEL_SRC).get_function("layernorm_kernel")
    return _kernel


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    """
    Apply Layer Normalization to each row of the input matrix.

    input : flattened row-major matrix, length divisible by row_size
    gamma, beta : vectors of length row_size
    Returns a flat float32 numpy array of the same size as input.
    """
    x = np.ascontiguousarray(input, dtype=np.float32).ravel()
    g = np.ascontiguousarray(gamma, dtype=np.float32).ravel()
    b = np.ascontiguousarray(beta, dtype=np.float32).ravel()
    row_size = int(row_size)

    if x.size == 0 or row_size <= 0:
        return np.empty(0, dtype=np.float32)

    row_count = x.size // row_size
    out = np.empty_like(x)

    d_x = drv.mem_alloc(x.nbytes)
    d_g = drv.mem_alloc(g.nbytes)
    d_b = drv.mem_alloc(b.nbytes)
    d_out = drv.mem_alloc(out.nbytes)
    try:
        drv.memcpy_htod(d_x, x)
        drv.memcpy_htod(d_g, g)
        drv.memcpy_htod(d_b, b)

        _get_kernel()(
            d_x, d_g, d_b, d_out,
            np.int32(row_size), np.float32(eps),
            block=(_BLOCK, 1, 1),
            grid=(row_count, 1),
            shared=_BLOCK * 4,
        )

        drv.memcpy_dtoh(out, d_out)
    finally:
        d_x.free()
        d_g.free()
        d_b.free()
        d_out.free()

    return out
