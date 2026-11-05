#include "softmax_cuda.h"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

void CheckCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

struct DeviceDeleter {
    void operator()(float* pointer) const noexcept {
        cudaFree(pointer);
    }
};

using DeviceBuffer = std::unique_ptr<float, DeviceDeleter>;

constexpr unsigned int kWarpSize = 32;
constexpr unsigned int kMaxThreads = 256;

__device__ float WarpMax(float value) {
    #pragma unroll
    for (unsigned int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

__device__ float WarpSum(float value) {
    #pragma unroll
    for (unsigned int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ float BlockMax(float value, float* partial) {
    const unsigned int lane = threadIdx.x % kWarpSize;
    const unsigned int warp = threadIdx.x / kWarpSize;
    value = WarpMax(value);
    if (blockDim.x == kWarpSize) {
        return __shfl_sync(0xffffffff, value, 0);
    }
    if (lane == 0) {
        partial[warp] = value;
    }
    __syncthreads();
    if (warp == 0) {
        value = lane < blockDim.x / kWarpSize ? partial[lane] : -CUDART_INF_F;
        value = WarpMax(value);
        if (lane == 0) {
            partial[0] = value;
        }
    }
    __syncthreads();
    return partial[0];
}

__device__ float BlockSum(float value, float* partial) {
    const unsigned int lane = threadIdx.x % kWarpSize;
    const unsigned int warp = threadIdx.x / kWarpSize;
    value = WarpSum(value);
    if (blockDim.x == kWarpSize) {
        return __shfl_sync(0xffffffff, value, 0);
    }
    if (lane == 0) {
        partial[warp] = value;
    }
    __syncthreads();
    if (warp == 0) {
        value = lane < blockDim.x / kWarpSize ? partial[lane] : 0.0f;
        value = WarpSum(value);
        if (lane == 0) {
            partial[0] = value;
        }
    }
    __syncthreads();
    return partial[0];
}

__global__ void SoftmaxKernel(const float* __restrict__ input,
                              float* __restrict__ output,
                              std::size_t row_size) {
    __shared__ float partial_max[kMaxThreads / kWarpSize];
    __shared__ float partial_sum[kMaxThreads / kWarpSize];

    const std::size_t row_offset = static_cast<std::size_t>(blockIdx.x) * row_size;
    float local_max = -CUDART_INF_F;
    for (std::size_t column = threadIdx.x; column < row_size; column += blockDim.x) {
        local_max = fmaxf(local_max, input[row_offset + column]);
    }
    const float row_max = BlockMax(local_max, partial_max);

    float local_sum = 0.0f;
    for (std::size_t column = threadIdx.x; column < row_size; column += blockDim.x) {
        const float value = __expf(input[row_offset + column] - row_max);
        output[row_offset + column] = value;
        local_sum += value;
    }
    const float inverse_sum = 1.0f / BlockSum(local_sum, partial_sum);

    for (std::size_t column = threadIdx.x; column < row_size; column += blockDim.x) {
        output[row_offset + column] *= inverse_sum;
    }
}

} // namespace

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (row_count < 0 || (row_count == 0 && !input.empty())) {
        throw std::invalid_argument("Row count must be positive for nonempty input");
    }
    if (input.empty()) {
        return {};
    }
    const std::size_t rows = static_cast<std::size_t>(row_count);
    if (input.size() % rows != 0) {
        throw std::invalid_argument("Input size must be divisible by row count");
    }
    const std::size_t row_size = input.size() / rows;
    if (input.size() > std::numeric_limits<std::size_t>::max() / (2 * sizeof(float))) {
        throw std::length_error("Input is too large");
    }

    const std::size_t bytes = input.size() * sizeof(float);
    float* storage = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&storage), 2 * bytes), "cudaMalloc");
    const DeviceBuffer device_storage(storage);
    float* const device_input = storage;
    float* const device_output = storage + input.size();
    CheckCuda(cudaMemcpy(device_input, input.data(), bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy input host to device");

    unsigned int threads = kWarpSize;
    while (threads < row_size && threads < kMaxThreads) {
        threads *= 2;
    }
    SoftmaxKernel<<<static_cast<unsigned int>(rows), threads>>>(
        device_input, device_output, row_size);
    CheckCuda(cudaGetLastError(), "SoftmaxKernel launch");

    std::vector<float> result(input.size());
    CheckCuda(cudaMemcpy(result.data(), device_output, bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy output device to host");
    return result;
}
