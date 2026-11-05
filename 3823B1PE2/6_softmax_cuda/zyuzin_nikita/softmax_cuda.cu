#include "softmax_cuda.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <thread>

namespace {

constexpr int kMaxThreads = 1024;
constexpr int kWarpSize = 32;

__device__ __forceinline__ float WarpMax(float value) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, offset));
    }
    return value;
}

__device__ __forceinline__ float WarpSum(float value) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_xor_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ float BlockMax(float value, float* shared) {
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    const int warps = blockDim.x / kWarpSize;
    value = WarpMax(value);
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();
    value = WarpMax(lane < warps ? shared[lane] : -INFINITY);
    __syncthreads();
    return value;
}

__device__ float BlockSum(float value, float* shared) {
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    const int warps = blockDim.x / kWarpSize;
    value = WarpSum(value);
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();
    value = WarpSum(lane < warps ? shared[lane] : 0.0f);
    __syncthreads();
    return value;
}

template <int kValuesPerThread>
__global__ void SoftmaxCachedKernel(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int row_size) {
    __shared__ float shared[kWarpSize];
    const size_t offset = static_cast<size_t>(blockIdx.x) * row_size;
    const float* in = input + offset;
    float* out = output + offset;

    float values[kValuesPerThread];
    float max_value = -INFINITY;
#pragma unroll
    for (int j = 0; j < kValuesPerThread; ++j) {
        const int i = threadIdx.x + j * blockDim.x;
        values[j] = i < row_size ? in[i] : -INFINITY;
        max_value = fmaxf(max_value, values[j]);
    }
    max_value = BlockMax(max_value, shared);

    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < kValuesPerThread; ++j) {
        values[j] = __expf(values[j] - max_value);
        sum += values[j];
    }
    const float inv_sum = 1.0f / BlockSum(sum, shared);

#pragma unroll
    for (int j = 0; j < kValuesPerThread; ++j) {
        const int i = threadIdx.x + j * blockDim.x;
        if (i < row_size) {
            out[i] = values[j] * inv_sum;
        }
    }
}

__global__ void SoftmaxKernel(const float* __restrict__ input,
                              float* __restrict__ output,
                              int row_size) {
    __shared__ float shared[kWarpSize];
    const size_t offset = static_cast<size_t>(blockIdx.x) * row_size;
    const float* in = input + offset;
    float* out = output + offset;

    float max_value = -INFINITY;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        max_value = fmaxf(max_value, in[i]);
    }
    max_value = BlockMax(max_value, shared);

    float sum = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        sum += __expf(in[i] - max_value);
    }
    const float inv_sum = 1.0f / BlockSum(sum, shared);

    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        out[i] = __expf(in[i] - max_value) * inv_sum;
    }
}

void LaunchSoftmax(const float* input, float* output, int row_count, int row_size) {
    int threads = row_size < kMaxThreads ? row_size : kMaxThreads;
    if (threads < kWarpSize) {
        threads = kWarpSize;
    }
    const int values_per_thread = (row_size + threads - 1) / threads;
    switch (values_per_thread) {
        case 1:
            SoftmaxCachedKernel<1><<<row_count, threads>>>(input, output, row_size);
            break;
        case 2:
            SoftmaxCachedKernel<2><<<row_count, threads>>>(input, output, row_size);
            break;
        case 4:
            SoftmaxCachedKernel<4><<<row_count, threads>>>(input, output, row_size);
            break;
        case 8:
            SoftmaxCachedKernel<8><<<row_count, threads>>>(input, output, row_size);
            break;
        case 16:
            SoftmaxCachedKernel<16><<<row_count, threads>>>(input, output, row_size);
            break;
        default:
            SoftmaxKernel<<<row_count, threads>>>(input, output, row_size);
            break;
    }
}

}  // namespace

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (row_count <= 0 || input.empty()) {
        return {};
    }
    const size_t count = input.size();
    const size_t bytes = count * sizeof(float);
    const int row_size = static_cast<int>(count / row_count);

    std::vector<float> output;
    std::thread allocator([&output, count] { output = std::vector<float>(count); });

    float* d_input = nullptr;
    float* d_output = nullptr;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);
    LaunchSoftmax(d_input, d_output, row_count, row_size);

    allocator.join();
    cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    return output;
}
