#include "softmax_cuda.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

struct MaxOp {
    __device__ __forceinline__ float operator()(float a, float b) const { return fmaxf(a, b); }
};

struct SumOp {
    __device__ __forceinline__ float operator()(float a, float b) const { return a + b; }
};

template <class Op>
__device__ __forceinline__ float BlockReduce(float val, float* shared, Op op) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = op(val, __shfl_xor_sync(0xffffffffu, val, offset));
    }

    if (threadIdx.x % 32 == 0) {
        shared[threadIdx.x / 32] = val;
    }
    __syncthreads();

    float result = shared[0];
#pragma unroll
    for (int i = 1; i < kWarps; ++i) {
        result = op(result, shared[i]);
    }
    __syncthreads();
    return result;
}

template <int kVecs>
__global__ void __launch_bounds__(kThreads)
SoftmaxCachedKernel(float* data, int row_size) {
    __shared__ float shared[kWarps];
    float4* row = reinterpret_cast<float4*>(data + static_cast<std::size_t>(blockIdx.x) * row_size);
    const int vecs = row_size / 4;

    float4 v[kVecs];
    float local_max = -INFINITY;

#pragma unroll
    for (int i = 0; i < kVecs; ++i) {
        const int j = threadIdx.x + i * kThreads;
        v[i] = (j < vecs) ? row[j] : make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
        local_max = fmaxf(local_max, fmaxf(fmaxf(v[i].x, v[i].y), fmaxf(v[i].z, v[i].w)));
    }
    const float row_max = BlockReduce(local_max, shared, MaxOp());

    float local_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecs; ++i) {
        v[i].x = __expf(v[i].x - row_max);
        v[i].y = __expf(v[i].y - row_max);
        v[i].z = __expf(v[i].z - row_max);
        v[i].w = __expf(v[i].w - row_max);
        local_sum += (v[i].x + v[i].y) + (v[i].z + v[i].w);
    }
    const float inv_sum = 1.0f / BlockReduce(local_sum, shared, SumOp());

#pragma unroll
    for (int i = 0; i < kVecs; ++i) {
        const int j = threadIdx.x + i * kThreads;
        if (j < vecs) {
            row[j] = make_float4(v[i].x * inv_sum, v[i].y * inv_sum, v[i].z * inv_sum, v[i].w * inv_sum);
        }
    }
}

__global__ void __launch_bounds__(kThreads)
SoftmaxGenericKernel(float* data, int row_size) {
    __shared__ float shared[kWarps];
    float* row = data + static_cast<std::size_t>(blockIdx.x) * row_size;

    float local_max = -INFINITY;
    for (int j = threadIdx.x; j < row_size; j += kThreads) {
        local_max = fmaxf(local_max, row[j]);
    }
    const float row_max = BlockReduce(local_max, shared, MaxOp());

    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < row_size; j += kThreads) {
        local_sum += __expf(row[j] - row_max);
    }
    const float inv_sum = 1.0f / BlockReduce(local_sum, shared, SumOp());

    for (int j = threadIdx.x; j < row_size; j += kThreads) {
        row[j] = __expf(row[j] - row_max) * inv_sum;
    }
}

struct DeviceMemory {
    float* ptr = nullptr;
    std::size_t capacity = 0;

    float* Acquire(std::size_t count) {
        if (count > capacity) {
            if (ptr != nullptr) {
                cudaFree(ptr);
            }
            cudaMalloc(&ptr, count * sizeof(float));
            capacity = count;
        }
        return ptr;
    }

    ~DeviceMemory() {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }
};

DeviceMemory g_device_mem;

}  // namespace

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (input.empty() || row_count <= 0) {
        return {};
    }

    const std::size_t count = input.size();
    const int row_size = static_cast<int>(count / row_count);
    const std::size_t bytes = count * sizeof(float);

    float* d_data = g_device_mem.Acquire(count);

    std::vector<float> output;
    std::thread host_alloc([&output, count] { output.resize(count); });

    cudaMemcpy(d_data, input.data(), bytes, cudaMemcpyHostToDevice);

    if (row_size % 4 == 0) {
        const int vecs = row_size / 4;
        if (vecs <= 256) SoftmaxCachedKernel<1><<<row_count, kThreads>>>(d_data, row_size);
        else if (vecs <= 512) SoftmaxCachedKernel<2><<<row_count, kThreads>>>(d_data, row_size);
        else if (vecs <= 1024) SoftmaxCachedKernel<4><<<row_count, kThreads>>>(d_data, row_size);
        else if (vecs <= 2048) SoftmaxCachedKernel<8><<<row_count, kThreads>>>(d_data, row_size);
        else if (vecs <= 4096) SoftmaxCachedKernel<16><<<row_count, kThreads>>>(d_data, row_size);
        else SoftmaxGenericKernel<<<row_count, kThreads>>>(d_data, row_size);
    } else {
        SoftmaxGenericKernel<<<row_count, kThreads>>>(d_data, row_size);
    }

    host_alloc.join();

    cudaMemcpy(output.data(), d_data, bytes, cudaMemcpyDeviceToHost);

    return output;
}
