#include "softmax_cuda.h"

#include <cuda_runtime.h>

#include <cfloat>
#include <cstddef>
#include <cstring>
#include <vector>

namespace {

constexpr int kWarpSize = 32;
constexpr int kMaxThreads = 256;

// Running (max, sum) pair for the online softmax algorithm. The sum is
// accumulated relative to the current maximum, so a single pass over a row is
// enough to get both the maximum and the sum of exponents.
struct MaxSum {
    float max;
    float sum;
};

__device__ __forceinline__ MaxSum MergeMaxSum(MaxSum a, MaxSum b) {
    const float m = fmaxf(a.max, b.max);
    if (m == -FLT_MAX) return {m, 0.0f};
    return {m, a.sum * __expf(a.max - m) + b.sum * __expf(b.max - m)};
}

__device__ __forceinline__ void AddElement(MaxSum& v, float x) {
    if (x > v.max) {
        v.sum = v.sum * __expf(v.max - x) + 1.0f;
        v.max = x;
    } else {
        v.sum += __expf(x - v.max);
    }
}

__device__ __forceinline__ MaxSum WarpReduce(MaxSum v) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        MaxSum other;
        other.max = __shfl_down_sync(0xffffffffu, v.max, offset);
        other.sum = __shfl_down_sync(0xffffffffu, v.sum, offset);
        v = MergeMaxSum(v, other);
    }
    return v;
}

// Block-wide reduction: first inside every warp with shuffles, then the first
// warp combines one partial result from each warp.
__device__ __forceinline__ MaxSum BlockReduce(MaxSum v, MaxSum* warp_results) {
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;

    v = WarpReduce(v);
    if (lane == 0) warp_results[warp] = v;
    __syncthreads();

    const int warp_count = blockDim.x / kWarpSize;
    MaxSum block = (threadIdx.x < warp_count) ? warp_results[threadIdx.x] : MaxSum{-FLT_MAX, 0.0f};
    if (warp == 0) block = WarpReduce(block);

    __shared__ MaxSum result;
    if (threadIdx.x == 0) result = block;
    __syncthreads();
    return result;
}

// One CUDA block processes one row. Rows whose length is a multiple of four are
// handled with float4 loads/stores, the remaining ones with the scalar path.
template <bool Vectorized>
__global__ void SoftmaxKernel(const float* __restrict__ input, float* __restrict__ output, int row_size) {
    __shared__ MaxSum warp_results[kMaxThreads / kWarpSize];

    const std::size_t base = static_cast<std::size_t>(blockIdx.x) * row_size;
    const float* row_in = input + base;
    float* row_out = output + base;

    MaxSum local{-FLT_MAX, 0.0f};
    if (Vectorized) {
        const int vec_count = row_size / 4;
        const float4* row_in4 = reinterpret_cast<const float4*>(row_in);
        for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
            const float4 v = row_in4[i];
            AddElement(local, v.x);
            AddElement(local, v.y);
            AddElement(local, v.z);
            AddElement(local, v.w);
        }
    } else {
        for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
            AddElement(local, row_in[i]);
        }
    }

    const MaxSum row = BlockReduce(local, warp_results);
    const float inv_sum = (row.sum > 0.0f) ? 1.0f / row.sum : 0.0f;

    if (Vectorized) {
        const int vec_count = row_size / 4;
        const float4* row_in4 = reinterpret_cast<const float4*>(row_in);
        float4* row_out4 = reinterpret_cast<float4*>(row_out);
        for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
            const float4 v = row_in4[i];
            const float4 r = make_float4(__expf(v.x - row.max) * inv_sum,
                                         __expf(v.y - row.max) * inv_sum,
                                         __expf(v.z - row.max) * inv_sum,
                                         __expf(v.w - row.max) * inv_sum);
            row_out4[i] = r;
        }
    } else {
        for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
            row_out[i] = __expf(row_in[i] - row.max) * inv_sum;
        }
    }
}

// Device buffers and a pinned host staging buffer are mapped once and reused,
// so no memory is allocated/freed during the measured runs.
struct Buffers {
    cudaStream_t stream = nullptr;
    float* dev_input = nullptr;
    float* dev_output = nullptr;
    float* host_output = nullptr;
    std::size_t capacity = 0;

    Buffers() { cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking); }

    ~Buffers() {
        cudaFree(dev_input);
        cudaFree(dev_output);
        if (host_output != nullptr) cudaFreeHost(host_output);
        if (stream != nullptr) cudaStreamDestroy(stream);
    }

    void Reserve(std::size_t count) {
        if (count <= capacity) return;
        cudaFree(dev_input);
        cudaFree(dev_output);
        if (host_output != nullptr) cudaFreeHost(host_output);

        cudaMalloc(&dev_input, count * sizeof(float));
        cudaMalloc(&dev_output, count * sizeof(float));
        cudaMallocHost(&host_output, count * sizeof(float));
        capacity = count;
    }
};

Buffers& GetBuffers() {
    static Buffers buffers;
    return buffers;
}

}  // namespace

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    const std::size_t count = input.size();
    if (count == 0 || row_count <= 0) return std::vector<float>(count);

    const std::size_t row_size = count / static_cast<std::size_t>(row_count);
    if (row_size == 0) return std::vector<float>(count);

    const std::size_t bytes = count * sizeof(float);
    Buffers& buf = GetBuffers();
    buf.Reserve(count);

    cudaMemcpyAsync(buf.dev_input, input.data(), bytes, cudaMemcpyHostToDevice, buf.stream);

    int block_size = kWarpSize;
    while (block_size < kMaxThreads && static_cast<std::size_t>(block_size) < row_size) {
        block_size <<= 1;
    }

    const bool vectorized = (row_size % 4 == 0);
    const int row_size_int = static_cast<int>(row_size);
    if (vectorized) {
        SoftmaxKernel<true><<<row_count, block_size, 0, buf.stream>>>(
            buf.dev_input, buf.dev_output, row_size_int);
    } else {
        SoftmaxKernel<false><<<row_count, block_size, 0, buf.stream>>>(
            buf.dev_input, buf.dev_output, row_size_int);
    }

    // The kernel is asynchronous, so the result vector is allocated on the host
    // while the GPU is still computing.
    std::vector<float> result(count);
    cudaMemcpyAsync(buf.host_output, buf.dev_output, bytes, cudaMemcpyDeviceToHost, buf.stream);
    cudaStreamSynchronize(buf.stream);
    std::memcpy(result.data(), buf.host_output, bytes);
    return result;
}
