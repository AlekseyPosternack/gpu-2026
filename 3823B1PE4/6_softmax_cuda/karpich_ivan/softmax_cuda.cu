#include "softmax_cuda.h"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstddef>
#include <future>
#include <stdexcept>

namespace {
constexpr int threads = 256;

void CheckCUDA(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

struct DeviceBuffer {
    float* data = nullptr;
    std::size_t capacity = 0;
    ~DeviceBuffer() { cudaFree(data); }
    void Reserve(std::size_t count) {
        if (count > capacity) {
            float* next = nullptr;
            CheckCUDA(cudaMalloc(&next, count * sizeof(float)));
            cudaFree(data);
            data = next;
            capacity = count;
        }
    }
};

template<bool Maximum>
__device__ float Reduce(float value) {
    __shared__ float partial[threads / 32];
    const int lane = threadIdx.x % 32;

#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        const float other = __shfl_down_sync(0xffffffffu, value, offset);
        value = Maximum ? fmaxf(value, other) : value + other;
    }
    if (lane == 0) {
        partial[threadIdx.x / 32] = value;
    }
    __syncthreads();

    value = threadIdx.x < threads / 32 ? partial[threadIdx.x] : (Maximum ? -CUDART_INF_F : 0.0f);
    if (threadIdx.x < 32) {
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            const float other = __shfl_down_sync(0xffffffffu, value, offset);
            value = Maximum ? fmaxf(value, other) : value + other;
        }
        if (lane == 0) {
            partial[0] = value;
        }
    }
    __syncthreads();
    return partial[0];
}

template<int Items>

__global__ void CachedSoftmaxKernel(float* data, std::size_t width) {
    float* row = data + static_cast<std::size_t>(blockIdx.x) * width;
    float values[Items];
    float maximum = -CUDART_INF_F;

#pragma unroll
    for (int i = 0; i < Items; ++i) {
        const std::size_t col = threadIdx.x + i * threads;
        values[i] = col < width ? row[col] : -CUDART_INF_F;
        maximum = fmaxf(maximum, values[i]);
    }
    maximum = Reduce<true>(maximum);
    float sum = 0.0f;

#pragma unroll
    for (int i = 0; i < Items; ++i) {
        values[i] = __expf(values[i] - maximum);
        sum += values[i];
    }
    const float inverse = 1.0f / Reduce<false>(sum);

#pragma unroll
    for (int i = 0; i < Items; ++i) {
        const std::size_t col = threadIdx.x + i * threads;
        if (col < width) {
            row[col] = values[i] * inverse;
        }
    }
}

__global__ void LongSoftmaxKernel(float* data, std::size_t width) {
    float* row = data + static_cast<std::size_t>(blockIdx.x) * width;
    float maximum = -CUDART_INF_F;
    for (std::size_t col = threadIdx.x; col < width; col += threads) {
        maximum = fmaxf(maximum, row[col]);
    }
    maximum = Reduce<true>(maximum);
    float sum = 0.0f;
    for (std::size_t col = threadIdx.x; col < width; col += threads) {
        sum += __expf(row[col] - maximum);
    }
    const float inverse = 1.0f / Reduce<false>(sum);
    for (std::size_t col = threadIdx.x; col < width; col += threads) {
        row[col] = __expf(row[col] - maximum) * inverse;
    }
}
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (row_count <= 0 || input.size() % row_count != 0) {
        throw std::invalid_argument("Invalid row count");
    }
    if (input.empty()) {
        return {};
    }
    const std::size_t count = input.size();
    const std::size_t width = count / row_count;
    const std::size_t bytes = count * sizeof(float);
    thread_local DeviceBuffer buffer;
    buffer.Reserve(count);
    auto output = std::async(std::launch::async, [count] { return std::vector<float>(count); });
    CheckCUDA(cudaMemcpy(buffer.data, input.data(), bytes, cudaMemcpyHostToDevice));
    if (width <= threads) {
        CachedSoftmaxKernel<1><<<row_count, threads>>>(buffer.data, width);
    } else if (width <= threads * 8) {
        CachedSoftmaxKernel<8><<<row_count, threads>>>(buffer.data, width);
    } else if (width <= threads * 32) {
        CachedSoftmaxKernel<32><<<row_count, threads>>>(buffer.data, width);
    } else if (width <= threads * 64) {
        CachedSoftmaxKernel<64><<<row_count, threads>>>(buffer.data, width);
    } else {
        LongSoftmaxKernel<<<row_count, threads>>>(buffer.data, width);
    }
    CheckCUDA(cudaGetLastError());
    std::vector<float> result = output.get();
    CheckCUDA(cudaMemcpy(result.data(), buffer.data, bytes, cudaMemcpyDeviceToHost));
    return result;
}
