#include "softmax_cuda.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cfloat>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int kMaxBlockThreads = 512;

constexpr int kMaxSharedCacheFloats = 8192;

constexpr int kMaxVecPerThread = 16;

__device__ __forceinline__ float WarpMaxXor(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    }
    return v;
}

__device__ __forceinline__ float WarpSumXor(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    return v;
}

__device__ __forceinline__ float BlockMaxXor(float v, float* red) {
    const int lane = threadIdx.x & 31;
    const int w    = threadIdx.x >> 5;
    const int num_warps = blockDim.x >> 5;

    v = WarpMaxXor(v);
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();

    v = (lane < num_warps) ? red[lane] : -FLT_MAX;
    return WarpMaxXor(v);
}

__device__ __forceinline__ float BlockSumXor(float v, float* red) {
    const int lane = threadIdx.x & 31;
    const int w    = threadIdx.x >> 5;
    const int num_warps = blockDim.x >> 5;

    v = WarpSumXor(v);
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();

    v = (lane < num_warps) ? red[lane] : 0.0f;
    return WarpSumXor(v);
}

template <bool kUseShared>
__global__ void softmax_kernel_impl(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int row_count,
                                    int row_size) {
    const int row = blockIdx.x;
    if (row >= row_count) return;

    const int tid = threadIdx.x;
    const float* row_in  = input  + static_cast<size_t>(row) * row_size;
    float*       row_out = output + static_cast<size_t>(row) * row_size;

    __shared__ float red[32];

    extern __shared__ float shared_row[];

    if (kUseShared) {
        float m = -FLT_MAX;
        for (int i = tid; i < row_size; i += blockDim.x) {
            const float v = row_in[i];
            shared_row[i] = v;
            m = fmaxf(m, v);
        }
        m = BlockMaxXor(m, red);

        float s = 0.0f;
        for (int i = tid; i < row_size; i += blockDim.x) {
            const float e = __expf(shared_row[i] - m);
            shared_row[i] = e;
            s += e;
        }
        s = BlockSumXor(s, red);

        const float inv = 1.0f / s;
        for (int i = tid; i < row_size; i += blockDim.x) {
            row_out[i] = shared_row[i] * inv;
        }
        return;
    }

    const int vec_per_thread = row_size / (blockDim.x * 4);
    float4 vals[kMaxVecPerThread];

    float local_max = -FLT_MAX;
    for (int i = 0; i < vec_per_thread; ++i) {
        const int vec_idx = i * blockDim.x + tid;
        float4 v = *reinterpret_cast<const float4*>(row_in + vec_idx * 4);
        vals[i] = v;
        local_max = fmaxf(local_max,
                          fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
    }

    const float row_max = BlockMaxXor(local_max, red);

    float local_sum = 0.0f;
    for (int i = 0; i < vec_per_thread; ++i) {
        float4 v = vals[i];
        v.x = __expf(v.x - row_max);
        v.y = __expf(v.y - row_max);
        v.z = __expf(v.z - row_max);
        v.w = __expf(v.w - row_max);
        vals[i] = v;
        local_sum += (v.x + v.y) + (v.z + v.w);
    }

    const float row_sum = BlockSumXor(local_sum, red);
    const float inv_sum = 1.0f / row_sum;

    for (int i = 0; i < vec_per_thread; ++i) {
        float4 v = vals[i];
        v.x *= inv_sum;
        v.y *= inv_sum;
        v.z *= inv_sum;
        v.w *= inv_sum;
        const int vec_idx = i * blockDim.x + tid;
        *reinterpret_cast<float4*>(row_out + vec_idx * 4) = v;
    }
}

float* g_d_in  = nullptr;
float* g_d_out = nullptr;
size_t g_capacity_bytes = 0;

void ReserveDeviceBuffers(size_t bytes) {
    if (bytes <= g_capacity_bytes && g_d_in && g_d_out) return;

    cudaDeviceSynchronize();
    if (g_d_in)  { cudaFree(g_d_in);  g_d_in  = nullptr; }
    if (g_d_out) { cudaFree(g_d_out); g_d_out = nullptr; }

    cudaMalloc(&g_d_in,  bytes);
    cudaMalloc(&g_d_out, bytes);
    g_capacity_bytes = bytes;
}

}  // namespace

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (input.empty() || row_count <= 0) return {};

    const size_t elems = input.size();
    const size_t bytes = elems * sizeof(float);
    const int row_size = static_cast<int>(elems / row_count);

    std::vector<float> output(elems);

    ReserveDeviceBuffers(bytes);

    cudaMemcpy(g_d_in, input.data(), bytes, cudaMemcpyHostToDevice);

    int block = 32;
    while (block < kMaxBlockThreads && block < row_size) block <<= 1;

    const bool use_shared = (row_size <= kMaxSharedCacheFloats);
    const size_t shared_bytes = use_shared
        ? static_cast<size_t>(row_size) * sizeof(float)
        : 0;

    dim3 grid(row_count);
    dim3 block_dim(block);

    if (use_shared) {
        softmax_kernel_impl<true><<<grid, block_dim, shared_bytes>>>(
            g_d_in, g_d_out, row_count, row_size);
    } else {
        softmax_kernel_impl<false><<<grid, block_dim, 0>>>(
            g_d_in, g_d_out, row_count, row_size);
    }

    cudaMemcpy(output.data(), g_d_out, bytes, cudaMemcpyDeviceToHost);

    return output;
}