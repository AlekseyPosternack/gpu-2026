#include "softmax_cuda.h"

#include <cuda_runtime.h>

#include <cfloat>
#include <cstddef>

namespace {

constexpr int kBlock = 256;

__device__ __forceinline__ float WarpReduceMax(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    }
    return v;
}

__device__ __forceinline__ float WarpReduceSum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__device__ __forceinline__ float BlockReduceMax(float v) {
    __shared__ float smem[32];
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    v = WarpReduceMax(v);
    if (lane == 0) {
        smem[wid] = v;
    }
    __syncthreads();
    v = (threadIdx.x < (blockDim.x >> 5)) ? smem[lane] : -FLT_MAX;
    if (wid == 0) {
        v = WarpReduceMax(v);
    }
    if (threadIdx.x == 0) {
        smem[0] = v;
    }
    __syncthreads();
    return smem[0];
}

__device__ __forceinline__ float BlockReduceSum(float v) {
    __shared__ float smem[32];
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    v = WarpReduceSum(v);
    if (lane == 0) {
        smem[wid] = v;
    }
    __syncthreads();
    v = (threadIdx.x < (blockDim.x >> 5)) ? smem[lane] : 0.0f;
    if (wid == 0) {
        v = WarpReduceSum(v);
    }
    if (threadIdx.x == 0) {
        smem[0] = v;
    }
    __syncthreads();
    return smem[0];
}

__global__ void SoftmaxScalarKernel(const float* __restrict__ in, float* __restrict__ out,
                                    int row_size) {
    const float* row = in + static_cast<std::size_t>(blockIdx.x) * row_size;
    float* y = out + static_cast<std::size_t>(blockIdx.x) * row_size;

    float local_max = -FLT_MAX;
    for (int i = threadIdx.x; i < row_size; i += kBlock) {
        local_max = fmaxf(local_max, row[i]);
    }
    local_max = BlockReduceMax(local_max);

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += kBlock) {
        const float e = __expf(row[i] - local_max);
        y[i] = e;
        local_sum += e;
    }
    local_sum = BlockReduceSum(local_sum);
    const float inv = 1.0f / local_sum;
    for (int i = threadIdx.x; i < row_size; i += kBlock) {
        y[i] *= inv;
    }
}

__global__ void SoftmaxKernel(const float* __restrict__ in, float* __restrict__ out, int row_size) {
    const float* row = in + static_cast<std::size_t>(blockIdx.x) * row_size;
    float* y = out + static_cast<std::size_t>(blockIdx.x) * row_size;
    const int n4 = row_size >> 2;

    float local_max = -FLT_MAX;
    for (int i = threadIdx.x; i < n4; i += kBlock) {
        const float4 v = reinterpret_cast<const float4*>(row)[i];
        local_max = fmaxf(local_max, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
    }
    local_max = BlockReduceMax(local_max);

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < n4; i += kBlock) {
        const float4 v = reinterpret_cast<const float4*>(row)[i];
        float4 e;
        e.x = __expf(v.x - local_max);
        e.y = __expf(v.y - local_max);
        e.z = __expf(v.z - local_max);
        e.w = __expf(v.w - local_max);
        reinterpret_cast<float4*>(y)[i] = e;
        local_sum += e.x + e.y + e.z + e.w;
    }
    local_sum = BlockReduceSum(local_sum);
    const float inv = 1.0f / local_sum;

    for (int i = threadIdx.x; i < n4; i += kBlock) {
        float4 e = reinterpret_cast<float4*>(y)[i];
        e.x *= inv;
        e.y *= inv;
        e.z *= inv;
        e.w *= inv;
        reinterpret_cast<float4*>(y)[i] = e;
    }
}

struct DeviceCache {
    float* d_in = nullptr;
    float* d_out = nullptr;
    cudaStream_t stream = nullptr;
    std::size_t cap = 0;
};

DeviceCache& Cache() {
    static DeviceCache cache;
    return cache;
}

void EnsureDevice(std::size_t n) {
    DeviceCache& cache = Cache();
    if (cache.stream == nullptr) {
        cudaStreamCreateWithFlags(&cache.stream, cudaStreamNonBlocking);
    }
    if (cache.cap >= n) {
        return;
    }
    if (cache.d_in != nullptr) {
        cudaFree(cache.d_in);
        cudaFree(cache.d_out);
    }
    const std::size_t bytes = n * sizeof(float);
    cudaMalloc(&cache.d_in, bytes);
    cudaMalloc(&cache.d_out, bytes);
    cache.cap = n;
}

}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (input.empty() || row_count <= 0) {
        return {};
    }

    const int n = static_cast<int>(input.size());
    const int row_size = n / row_count;
    EnsureDevice(static_cast<std::size_t>(n));
    DeviceCache& cache = Cache();
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);

    cudaMemcpyAsync(cache.d_in, input.data(), bytes, cudaMemcpyHostToDevice, cache.stream);

    std::vector<float> output(static_cast<std::size_t>(n));

    if ((row_size & 3) == 0) {
        SoftmaxKernel<<<row_count, kBlock, 0, cache.stream>>>(cache.d_in, cache.d_out, row_size);
    } else {
        SoftmaxScalarKernel<<<row_count, kBlock, 0, cache.stream>>>(cache.d_in, cache.d_out,
                                                                    row_size);
    }

    cudaMemcpyAsync(output.data(), cache.d_out, bytes, cudaMemcpyDeviceToHost, cache.stream);
    cudaStreamSynchronize(cache.stream);
    return output;
}
