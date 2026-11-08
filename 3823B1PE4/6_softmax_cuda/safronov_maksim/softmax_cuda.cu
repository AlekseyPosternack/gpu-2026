#include "softmax_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace {

constexpr int kBlockSize = 256;
constexpr int kWarpSize = 32;

void CheckCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

__device__ __forceinline__ float WarpMax(float value) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

__device__ __forceinline__ float WarpSum(float value) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ __forceinline__ float BlockMax(float value, float* partial) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    value = WarpMax(value);
    if (lane == 0) {
        partial[warp] = value;
    }
    __syncthreads();
    if (warp == 0) {
        value = lane < kBlockSize / kWarpSize ? partial[lane] : -3.402823466e+38F;
        value = WarpMax(value);
        if (lane == 0) {
            partial[0] = value;
        }
    }
    __syncthreads();
    return partial[0];
}

__device__ __forceinline__ float BlockSum(float value, float* partial) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    value = WarpSum(value);
    if (lane == 0) {
        partial[warp] = value;
    }
    __syncthreads();
    if (warp == 0) {
        value = lane < kBlockSize / kWarpSize ? partial[lane] : 0.0f;
        value = WarpSum(value);
        if (lane == 0) {
            partial[0] = value;
        }
    }
    __syncthreads();
    return partial[0];
}

__global__ void SoftmaxKernel(float* data, std::size_t rowSize) {
    __shared__ float partial[kWarpSize];
    const std::size_t rowStart = static_cast<std::size_t>(blockIdx.x) * rowSize;

    float maximum = -3.402823466e+38F;
    for (std::size_t column = threadIdx.x; column < rowSize; column += kBlockSize) {
        maximum = fmaxf(maximum, data[rowStart + column]);
    }
    maximum = BlockMax(maximum, partial);

    float sum = 0.0f;
    for (std::size_t column = threadIdx.x; column < rowSize; column += kBlockSize) {
        const float value = __expf(data[rowStart + column] - maximum);
        data[rowStart + column] = value;
        sum += value;
    }
    sum = BlockSum(sum, partial);

    const float inverse = __fdividef(1.0f, sum);
    for (std::size_t column = threadIdx.x; column < rowSize; column += kBlockSize) {
        data[rowStart + column] *= inverse;
    }
}

class DeviceBuffer {
public:
    ~DeviceBuffer() {
        cudaFree(data_);
    }

    float* Reserve(std::size_t size) {
        if (size <= capacity_) {
            return data_;
        }
        if (data_ != nullptr) {
            CheckCuda(cudaFree(data_), "cudaFree");
            data_ = nullptr;
            capacity_ = 0;
        }
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&data_), size * sizeof(float)), "cudaMalloc");
        capacity_ = size;
        return data_;
    }

private:
    float* data_ = nullptr;
    std::size_t capacity_ = 0;
};

}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (row_count <= 0) {
        throw std::invalid_argument("row count must be positive");
    }
    if (input.size() % static_cast<std::size_t>(row_count) != 0) {
        throw std::invalid_argument("input size must be divisible by row count");
    }
    if (input.empty()) {
        return {};
    }

    const std::size_t rowSize = input.size() / static_cast<std::size_t>(row_count);
    const std::size_t bytes = input.size() * sizeof(float);
    static thread_local DeviceBuffer buffer;
    float* deviceData = buffer.Reserve(input.size());

    CheckCuda(cudaMemcpy(deviceData, input.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy input");
    SoftmaxKernel<<<static_cast<unsigned int>(row_count), kBlockSize>>>(deviceData, rowSize);
    CheckCuda(cudaGetLastError(), "SoftmaxKernel launch");

    std::vector<float> result(input.size());
    CheckCuda(cudaMemcpy(result.data(), deviceData, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy result");
    return result;
}
