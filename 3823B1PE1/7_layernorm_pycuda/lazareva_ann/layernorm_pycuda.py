import numpy as np
import pycuda.autoinit  
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


_CUDA_SOURCE = r"""
#define WARP_SIZE 32

// Block-level sum reduction via warp-shuffle intrinsics.
// warpBuf must have at least (blockDim.x / WARP_SIZE) elements.
__device__ __forceinline__ float BlockReduceSum(float val, float* warpBuf) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid  = threadIdx.x / WARP_SIZE;
    int numWarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if (lane == 0) warpBuf[wid] = val;
    __syncthreads();

    val = (threadIdx.x < numWarps) ? warpBuf[threadIdx.x] : 0.0f;
    if (wid == 0) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0) warpBuf[0] = val;
    }
    __syncthreads();
    return warpBuf[0];
}

// Main kernel: the whole row is cached in dynamic shared memory.
// One CUDA block processes one row.
extern "C" __global__
void LayerNormKernelShared(const float* __restrict__ input,
                            const float* __restrict__ gamma,
                            const float* __restrict__ beta,
                            float* __restrict__ output,
                            int row_size, float eps) {
    extern __shared__ float srow[];
    __shared__ float warpBuf[32];

    int row = blockIdx.x;
    const float* in  = input  + (size_t)row * row_size;
    float*       out = output + (size_t)row * row_size;

    int tid    = threadIdx.x;
    int stride = blockDim.x;

    float localSum = 0.0f;
    for (int i = tid; i < row_size; i += stride) {
        float v = in[i];
        srow[i] = v;
        localSum += v;
    }
    __syncthreads();

    float sum  = BlockReduceSum(localSum, warpBuf);
    float mean = sum / row_size;

    float localSumSq = 0.0f;
    for (int i = tid; i < row_size; i += stride) {
        float d = srow[i] - mean;
        localSumSq += d * d;
    }
    float sumSq = BlockReduceSum(localSumSq, warpBuf);
    float var   = sumSq / row_size;
    float invStd = rsqrtf(var + eps);

    for (int i = tid; i < row_size; i += stride) {
        float norm = (srow[i] - mean) * invStd;
        out[i] = norm * __ldg(&gamma[i]) + __ldg(&beta[i]);
    }
}

// Fallback kernel (no shared-memory row caching) for rows that do not
// fit into the available shared memory. Three passes over global
// memory, but access is still fully coalesced.
extern "C" __global__
void LayerNormKernelGlobal(const float* __restrict__ input,
                            const float* __restrict__ gamma,
                            const float* __restrict__ beta,
                            float* __restrict__ output,
                            int row_size, float eps) {
    __shared__ float warpBuf[32];

    int row = blockIdx.x;
    const float* in  = input  + (size_t)row * row_size;
    float*       out = output + (size_t)row * row_size;

    int tid    = threadIdx.x;
    int stride = blockDim.x;

    float localSum = 0.0f;
    for (int i = tid; i < row_size; i += stride) localSum += in[i];
    float sum  = BlockReduceSum(localSum, warpBuf);
    float mean = sum / row_size;

    float localSumSq = 0.0f;
    for (int i = tid; i < row_size; i += stride) {
        float d = in[i] - mean;
        localSumSq += d * d;
    }
    float sumSq = BlockReduceSum(localSumSq, warpBuf);
    float var   = sumSq / row_size;
    float invStd = rsqrtf(var + eps);

    for (int i = tid; i < row_size; i += stride) {
        float norm = (in[i] - mean) * invStd;
        out[i] = norm * __ldg(&gamma[i]) + __ldg(&beta[i]);
    }
}
"""


class _LayerNormContext:
    """Holds persistent CUDA resources: compiled kernels, a stream, and
    device/pinned-host buffers. Created once (lazily, on first call) and
    reused across subsequent calls to layernorm_pycuda, avoiding repeated
    compilation and allocation overhead."""

    def __init__(self):
        module = SourceModule(_CUDA_SOURCE, options=["--use_fast_math"])
        self.shared_kernel = module.get_function("LayerNormKernelShared")
        self.global_kernel = module.get_function("LayerNormKernelGlobal")
        self.stream = cuda.Stream()

        device = cuda.Context.get_device()
        default_max_shared = device.get_attribute(
            cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK)

        try:
            optin_max_shared = device.get_attribute(
                cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
            self.shared_kernel.set_attribute(
                cuda.function_attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES,
                optin_max_shared)
            self.max_shared_per_block = optin_max_shared
        except Exception:
            self.max_shared_per_block = default_max_shared

        self.block_dim_cap = 256

        self.d_in = None
        self.d_out = None
        self.data_capacity = 0

        self.d_gamma = None
        self.d_beta = None
        self.param_capacity = 0

        self.h_in = None
        self.h_out = None
        self.h_gamma = None
        self.h_beta = None

    def ensure_data_capacity(self, n):
        if n > self.data_capacity:
            if self.d_in is not None:
                self.d_in.free()
                self.d_out.free()
            self.d_in = cuda.mem_alloc(n * 4)
            self.d_out = cuda.mem_alloc(n * 4)
            self.h_in = cuda.pagelocked_empty(n, dtype=np.float32)
            self.h_out = cuda.pagelocked_empty(n, dtype=np.float32)
            self.data_capacity = n

    def ensure_param_capacity(self, row_size):
        if row_size > self.param_capacity:
            if self.d_gamma is not None:
                self.d_gamma.free()
                self.d_beta.free()
            self.d_gamma = cuda.mem_alloc(row_size * 4)
            self.d_beta = cuda.mem_alloc(row_size * 4)
            self.h_gamma = cuda.pagelocked_empty(row_size, dtype=np.float32)
            self.h_beta = cuda.pagelocked_empty(row_size, dtype=np.float32)
            self.param_capacity = row_size


_ctx = None


def _get_ctx():
    global _ctx
    if _ctx is None:
        _ctx = _LayerNormContext()
    return _ctx


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    """
    Apply Layer Normalization to each row of the input matrix.

    Parameters
    ----------
    input : list or numpy.ndarray of float
        Flattened matrix in row-major order. Its length must be divisible
        by row_size.
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
        Flattened matrix of the same shape as input, containing the
        row-wise normalized results.
    """
    ctx = _get_ctx()
    stream = ctx.stream

    input_np = np.ascontiguousarray(input, dtype=np.float32).ravel()
    gamma_np = np.ascontiguousarray(gamma, dtype=np.float32).ravel()
    beta_np = np.ascontiguousarray(beta, dtype=np.float32).ravel()

    total = input_np.size
    if total == 0:
        return np.array([], dtype=np.float32)

    row_count = total // row_size

    ctx.ensure_data_capacity(total)
    ctx.ensure_param_capacity(row_size)

    ctx.h_in[:total] = input_np
    ctx.h_gamma[:row_size] = gamma_np
    ctx.h_beta[:row_size] = beta_np

    cuda.memcpy_htod_async(ctx.d_in, ctx.h_in[:total], stream)
    cuda.memcpy_htod_async(ctx.d_gamma, ctx.h_gamma[:row_size], stream)
    cuda.memcpy_htod_async(ctx.d_beta, ctx.h_beta[:row_size], stream)

    block = min(row_size, ctx.block_dim_cap)
    shared_bytes = row_size * 4

    if shared_bytes <= ctx.max_shared_per_block:
        ctx.shared_kernel(
            ctx.d_in, ctx.d_gamma, ctx.d_beta, ctx.d_out,
            np.int32(row_size), np.float32(eps),
            block=(block, 1, 1), grid=(row_count, 1),
            shared=shared_bytes, stream=stream)
    else:
        ctx.global_kernel(
            ctx.d_in, ctx.d_gamma, ctx.d_beta, ctx.d_out,
            np.int32(row_size), np.float32(eps),
            block=(block, 1, 1), grid=(row_count, 1),
            stream=stream)

    cuda.memcpy_dtoh_async(ctx.h_out[:total], ctx.d_out, stream)
    stream.synchronize()

    return np.array(ctx.h_out[:total])