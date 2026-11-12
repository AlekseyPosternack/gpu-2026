import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

CUDA_KERNEL_SOURCE = """
#define WARP_SIZE 32

// Warp-level sum reduction using registers
__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level sum reduction across warps using shared memory
__inline__ __device__ float blockReduceSum(float val) {
    __shared__ float shared[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warpReduceSum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    int num_warps = blockDim.x / WARP_SIZE;
    val = (threadIdx.x < num_warps) ? shared[lane] : 0.0f;

    if (wid == 0) {
        val = warpReduceSum(val);
        if (threadIdx.x == 0) {
            shared[0] = val;
        }
    }
    __syncthreads();

    return shared[0];
}

// Helper that handles both single-warp (blockDim.x <= 32) and multi-warp blocks
__inline__ __device__ float reduceSum(float val) {
    if (blockDim.x <= WARP_SIZE) {
        val = warpReduceSum(val);
        return __shfl_sync(0xffffffff, val, 0);
    }
    return blockReduceSum(val);
}

// Kernel 1: Caches the row in shared memory (for row_size <= 4096)
extern "C" __global__ void layernormSharedMemKernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int row_size,
    float eps)
{
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    size_t row_offset = static_cast<size_t>(row) * row_size;
    const float* row_in = input + row_offset;
    float* row_out = output + row_offset;

    // 1. Load row into shared memory and compute thread-local sum
    float local_sum = 0.0f;
    for (int i = tid; i < row_size; i += blockDim.x) {
        float val = row_in[i];
        sdata[i] = val;
        local_sum += val;
    }

    float mean = reduceSum(local_sum) / static_cast<float>(row_size);

    // 2. Center values in-place in shared memory and compute variance sum
    float local_var_sum = 0.0f;
    for (int i = tid; i < row_size; i += blockDim.x) {
        float diff = sdata[i] - mean;
        sdata[i] = diff;
        local_var_sum += diff * diff;
    }

    float var = reduceSum(local_var_sum) / static_cast<float>(row_size);
    float inv_std = rsqrtf(var + eps);

    // 3. Normalize, scale by gamma, shift by beta, and write to global memory
    for (int i = tid; i < row_size; i += blockDim.x) {
        row_out[i] = (sdata[i] * inv_std) * __ldg(&gamma[i]) + __ldg(&beta[i]);
    }
}

// Kernel 2: Vectorized (float4) kernel for large rows (> 4096 elements)
extern "C" __global__ void layernormLargeRowKernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int row_size,
    float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    size_t row_offset = static_cast<size_t>(row) * row_size;

    if (row_size >= 4) {
        const float4* row_in4 = reinterpret_cast<const float4*>(input + row_offset);
        const float4* gamma4 = reinterpret_cast<const float4*>(gamma);
        const float4* beta4 = reinterpret_cast<const float4*>(beta);
        float4* row_out4 = reinterpret_cast<float4*>(output + row_offset);
        int vec_size = row_size / 4;

        // 1. Compute mean using float4 loads
        float local_sum = 0.0f;
        for (int i = tid; i < vec_size; i += blockDim.x) {
            float4 v = row_in4[i];
            local_sum += (v.x + v.y) + (v.z + v.w);
        }
        float mean = reduceSum(local_sum) / static_cast<float>(row_size);

        // 2. Compute variance using float4 loads
        float local_var_sum = 0.0f;
        for (int i = tid; i < vec_size; i += blockDim.x) {
            float4 v = row_in4[i];
            float dx = v.x - mean;
            float dy = v.y - mean;
            float dz = v.z - mean;
            float dw = v.w - mean;
            local_var_sum += (dx * dx + dy * dy) + (dz * dz + dw * dw);
        }
        float var = reduceSum(local_var_sum) / static_cast<float>(row_size);
        float inv_std = rsqrtf(var + eps);

        // 3. Normalize, scale, shift, and write float4 back
        for (int i = tid; i < vec_size; i += blockDim.x) {
            float4 v = row_in4[i];
            float4 g = __ldg(&gamma4[i]);
            float4 b = __ldg(&beta4[i]);
            float4 out;
            out.x = ((v.x - mean) * inv_std) * g.x + b.x;
            out.y = ((v.y - mean) * inv_std) * g.y + b.y;
            out.z = ((v.z - mean) * inv_std) * g.z + b.z;
            out.w = ((v.w - mean) * inv_std) * g.w + b.w;
            row_out4[i] = out;
        }
    } else {
        const float* row_in = input + row_offset;
        float* row_out = output + row_offset;

        float local_sum = 0.0f;
        for (int i = tid; i < row_size; i += blockDim.x) {
            local_sum += row_in[i];
        }
        float mean = reduceSum(local_sum) / static_cast<float>(row_size);

        float local_var_sum = 0.0f;
        for (int i = tid; i < row_size; i += blockDim.x) {
            float diff = row_in[i] - mean;
            local_var_sum += diff * diff;
        }
        float var = reduceSum(local_var_sum) / static_cast<float>(row_size);
        float inv_std = rsqrtf(var + eps);

        for (int i = tid; i < row_size; i += blockDim.x) {
            row_out[i] = ((row_in[i] - mean) * inv_std) * __ldg(&gamma[i]) + __ldg(&beta[i]);
        }
    }
}
"""

_KERNELS = None


def _get_kernels():
    """Compile and cache the CUDA kernels once per context."""
    global _KERNELS
    if _KERNELS is None:
        mod = SourceModule(CUDA_KERNEL_SOURCE, options=["-O3"], no_extern_c=True)
        _KERNELS = (
            mod.get_function("layernormSharedMemKernel"),
            mod.get_function("layernormLargeRowKernel"),
        )
    return _KERNELS


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
    orig_shape = np.shape(input)
    x = np.ascontiguousarray(input, dtype=np.float32).ravel()
    if x.size == 0 or row_size <= 0:
        return x.reshape(orig_shape)

    g = np.ascontiguousarray(gamma, dtype=np.float32).ravel()
    b = np.ascontiguousarray(beta, dtype=np.float32).ravel()

    row_count = x.size // row_size
    shared_kernel, large_kernel = _get_kernels()

    stream = cuda.Stream()

    # Allocate device memory
    d_x = cuda.mem_alloc(x.nbytes)
    d_g = cuda.mem_alloc(g.nbytes)
    d_b = cuda.mem_alloc(b.nbytes)
    d_out = cuda.mem_alloc(x.nbytes)

    # Asynchronously transfer inputs to GPU
    cuda.memcpy_htod_async(d_x, x, stream)
    cuda.memcpy_htod_async(d_g, g, stream)
    cuda.memcpy_htod_async(d_b, b, stream)

    # Launch the appropriate kernel
    if row_size <= 4096:
        threads = min(512, max(32, row_size))
        shared_bytes = row_size * 4
        shared_kernel(
            d_x,
            d_g,
            d_b,
            d_out,
            np.int32(row_size),
            np.float32(eps),
            block=(threads, 1, 1),
            grid=(row_count, 1, 1),
            shared=shared_bytes,
            stream=stream,
        )
    else:
        threads = 256
        large_kernel(
            d_x,
            d_g,
            d_b,
            d_out,
            np.int32(row_size),
            np.float32(eps),
            block=(threads, 1, 1),
            grid=(row_count, 1, 1),
            shared=0,
            stream=stream,
        )

    # Overlap host output buffer allocation while GPU executes
    out = np.empty_like(x)

    # Copy result back and synchronize
    cuda.memcpy_dtoh_async(out, d_out, stream)
    stream.synchronize()

    # Explicitly release device buffers
    d_x.free()
    d_g.free()
    d_b.free()
    d_out.free()

    return out.reshape(orig_shape)