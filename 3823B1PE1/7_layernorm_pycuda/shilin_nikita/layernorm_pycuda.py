import numpy as np

_KERNEL = r"""
__device__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__device__ float block_sum(float v) {
    __shared__ float smem[32];
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) {
        smem[wid] = v;
    }
    __syncthreads();
    v = (threadIdx.x < (blockDim.x >> 5)) ? smem[lane] : 0.0f;
    if (wid == 0) {
        v = warp_sum(v);
    }
    if (threadIdx.x == 0) {
        smem[0] = v;
    }
    __syncthreads();
    return smem[0];
}

__global__ void layernorm_row(const float* __restrict__ x,
                              const float* __restrict__ gamma,
                              const float* __restrict__ beta,
                              float* __restrict__ y,
                              int row_size,
                              float eps) {
    const float* row = x + (size_t)blockIdx.x * row_size;
    float* out = y + (size_t)blockIdx.x * row_size;
    const int n4 = row_size >> 2;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        const float4 v = reinterpret_cast<const float4*>(row)[i];
        sum += v.x + v.y + v.z + v.w;
    }
    const float mean = block_sum(sum) / (float)row_size;

    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        const float4 v = reinterpret_cast<const float4*>(row)[i];
        const float d0 = v.x - mean;
        const float d1 = v.y - mean;
        const float d2 = v.z - mean;
        const float d3 = v.w - mean;
        var_sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }
    const float inv = rsqrtf(block_sum(var_sum) / (float)row_size + eps);

    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        const float4 v = reinterpret_cast<const float4*>(row)[i];
        const float4 g = reinterpret_cast<const float4*>(gamma)[i];
        const float4 b = reinterpret_cast<const float4*>(beta)[i];
        float4 r;
        r.x = g.x * (v.x - mean) * inv + b.x;
        r.y = g.y * (v.y - mean) * inv + b.y;
        r.z = g.z * (v.z - mean) * inv + b.z;
        r.w = g.w * (v.w - mean) * inv + b.w;
        reinterpret_cast<float4*>(out)[i] = r;
    }
}

__global__ void layernorm_row_scalar(const float* __restrict__ x,
                                     const float* __restrict__ gamma,
                                     const float* __restrict__ beta,
                                     float* __restrict__ y,
                                     int row_size,
                                     float eps) {
    const float* row = x + (size_t)blockIdx.x * row_size;
    float* out = y + (size_t)blockIdx.x * row_size;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        sum += row[i];
    }
    const float mean = block_sum(sum) / (float)row_size;

    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        const float d = row[i] - mean;
        var_sum += d * d;
    }
    const float inv = rsqrtf(block_sum(var_sum) / (float)row_size + eps);

    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        out[i] = gamma[i] * (row[i] - mean) * inv + beta[i];
    }
}
"""

_mod = None
_fn = None
_fn_scalar = None
_gpu = None


def _module():
    global _mod, _fn, _fn_scalar
    if _fn is not None:
        return _fn, _fn_scalar

    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule

    _mod = SourceModule(_KERNEL, options=["-O3", "--use_fast_math"])
    _fn = _mod.get_function("layernorm_row")
    _fn_scalar = _mod.get_function("layernorm_row_scalar")
    return _fn, _fn_scalar


class _GpuBufs:
    def __init__(self):
        self.n = 0
        self.row_size = 0
        self.d_x = None
        self.d_y = None
        self.d_g = None
        self.d_b = None


def _bufs():
    global _gpu
    if _gpu is None:
        _gpu = _GpuBufs()
    return _gpu


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
    import pycuda.driver as cuda

    x = np.ascontiguousarray(input, dtype=np.float32).ravel()
    g = np.ascontiguousarray(gamma, dtype=np.float32).ravel()
    b = np.ascontiguousarray(beta, dtype=np.float32).ravel()
    row_size = int(row_size)
    n = int(x.size)
    rows = n // row_size

    fn, fn_scalar = _module()
    gpu = _bufs()
    if gpu.n < n:
        gpu.d_x = cuda.mem_alloc(n * np.float32().nbytes)
        gpu.d_y = cuda.mem_alloc(n * np.float32().nbytes)
        gpu.n = n
    if gpu.row_size < row_size:
        gpu.d_g = cuda.mem_alloc(row_size * np.float32().nbytes)
        gpu.d_b = cuda.mem_alloc(row_size * np.float32().nbytes)
        gpu.row_size = row_size

    cuda.memcpy_htod(gpu.d_x, x)
    cuda.memcpy_htod(gpu.d_g, g)
    cuda.memcpy_htod(gpu.d_b, b)

    block = 256
    kernel = fn if (row_size & 3) == 0 else fn_scalar
    kernel(
        gpu.d_x,
        gpu.d_g,
        gpu.d_b,
        gpu.d_y,
        np.int32(row_size),
        np.float32(eps),
        block=(block, 1, 1),
        grid=(rows, 1),
    )

    out = np.empty(n, dtype=np.float32)
    cuda.memcpy_dtoh(out, gpu.d_y)
    return out
