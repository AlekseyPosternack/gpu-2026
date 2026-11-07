#include "softmax_cuda.h"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

constexpr int ThreadsPerBlock = 256;
constexpr std::size_t MaxBlocks = 65535;

void CheckCudaError(const cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(error)
        );
    }
}

struct DeviceMemoryDeleter {
    void operator()(float* pointer) const noexcept {
        cudaFree(pointer);
    }
};

template <int BlockSize, int ItemsPerThread>
__global__ void SoftmaxKernel(const float* __restrict__ input,
                              float* __restrict__ output,
                              const std::size_t rowCount,
                              const std::size_t rowSize) {
    __shared__ float reduction[BlockSize];
    const unsigned int tid = threadIdx.x;
    float values[ItemsPerThread > 0 ? ItemsPerThread : 1];

    for (std::size_t row = blockIdx.x; row < rowCount; row += gridDim.x) {
        const std::size_t base = row * rowSize;
        float localMax = -CUDART_INF_F;

        if constexpr (ItemsPerThread > 0) {

            #pragma unroll
            for (int item = 0; item < ItemsPerThread; ++item) {
                const std::size_t column =
                    tid + static_cast<std::size_t>(item) * BlockSize;
                values[item] = column < rowSize
                    ? input[base + column] : -CUDART_INF_F;
                localMax = fmaxf(localMax, values[item]);
            }
        } else {
            for (std::size_t column = tid; column < rowSize; column += BlockSize) {
                localMax = fmaxf(localMax, input[base + column]);
            }
        }

        reduction[tid] = localMax;
        __syncthreads();
        #pragma unroll
        for (int stride = BlockSize / 2; stride > 0; stride /= 2) {
            if (tid < static_cast<unsigned int>(stride)) {
                reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
            }
            __syncthreads();
        }
        const float rowMax = reduction[0];
        __syncthreads();

        float localSum = 0.0f;
        if constexpr (ItemsPerThread > 0) {
            #pragma unroll
            for (int item = 0; item < ItemsPerThread; ++item) {
                values[item] = expf(values[item] - rowMax);
                localSum += values[item];
            }
        } else {
            for (std::size_t column = tid; column < rowSize; column += BlockSize) {
                const float value = expf(input[base + column] - rowMax);
                output[base + column] = value;
                localSum += value;
            }
        }

        reduction[tid] = localSum;
        __syncthreads();
        #pragma unroll
        for (int stride = BlockSize / 2; stride > 0; stride /= 2) {
            if (tid < static_cast<unsigned int>(stride)) {
                reduction[tid] += reduction[tid + stride];
            }
            __syncthreads();
        }
        const float inverseSum = 1.0f / reduction[0];
        __syncthreads();

        if constexpr (ItemsPerThread > 0) {
            #pragma unroll
            for (int item = 0; item < ItemsPerThread; ++item) {
                const std::size_t column =
                    tid + static_cast<std::size_t>(item) * BlockSize;
                if (column < rowSize) {
                    output[base + column] = values[item] * inverseSum;
                }
            }
        } else {
            for (std::size_t column = tid; column < rowSize; column += BlockSize) {
                output[base + column] *= inverseSum;
            }
        }
    }
}

template <int BlockSize, int ItemsPerThread>
void LaunchSoftmax(const float* input, float* output,
                   const std::size_t rowCount, const std::size_t rowSize) {
    const unsigned int blockCount = static_cast<unsigned int>(
        rowCount < MaxBlocks ? rowCount : MaxBlocks
    );
    SoftmaxKernel<BlockSize, ItemsPerThread><<<blockCount, BlockSize>>>(
        input, output, rowCount, rowSize
    );
}

void DispatchSoftmax(const float* input, float* output,
                     const std::size_t rowCount, const std::size_t rowSize) {
    if (rowSize <= 32) {
        LaunchSoftmax<32, 1>(input, output, rowCount, rowSize);
    } else if (rowSize <= 64) {
        LaunchSoftmax<64, 1>(input, output, rowCount, rowSize);
    } else if (rowSize <= 128) {
        LaunchSoftmax<128, 1>(input, output, rowCount, rowSize);
    } else if (rowSize <= 256) {
        LaunchSoftmax<ThreadsPerBlock, 1>(input, output, rowCount, rowSize);
    } else if (rowSize <= 512) {
        LaunchSoftmax<ThreadsPerBlock, 2>(input, output, rowCount, rowSize);
    } else if (rowSize <= 1024) {
        LaunchSoftmax<ThreadsPerBlock, 4>(input, output, rowCount, rowSize);
    } else if (rowSize <= 2048) {
        LaunchSoftmax<ThreadsPerBlock, 8>(input, output, rowCount, rowSize);
    } else if (rowSize <= 4096) {
        LaunchSoftmax<ThreadsPerBlock, 16>(input, output, rowCount, rowSize);
    } else if (rowSize <= 8192) {
        LaunchSoftmax<ThreadsPerBlock, 32>(input, output, rowCount, rowSize);
    } else if (rowSize <= 16384) {
        LaunchSoftmax<ThreadsPerBlock, 64>(input, output, rowCount, rowSize);
    } else {
        LaunchSoftmax<ThreadsPerBlock, 0>(input, output, rowCount, rowSize);
    }
}

} // namespace

std::vector<float> SoftmaxCUDA(const std::vector<float>& input,
                              const int row_count) {
    if (row_count < 0) {
        throw std::invalid_argument("Row count must be non-negative");
    }
    if (input.empty()) {
        return {};
    }
    if (row_count == 0 ||
        input.size() % static_cast<std::size_t>(row_count) != 0) {
        throw std::invalid_argument("Input size must be divisible by a positive row count");
    }

    const std::size_t elementCount = input.size();
    if (elementCount > std::numeric_limits<std::size_t>::max() / sizeof(float) / 2) {
        throw std::length_error("Device allocation size is too large");
    }
    const std::size_t rowCount = static_cast<std::size_t>(row_count);
    const std::size_t rowSize = elementCount / rowCount;
    const std::size_t bytes = elementCount * sizeof(float);

    void* allocation = nullptr;
    CheckCudaError(cudaMalloc(&allocation, 2 * bytes),
                   "Failed to allocate device input and output");
    const std::unique_ptr<float, DeviceMemoryDeleter> deviceMemory(
        static_cast<float*>(allocation)
    );
    float* const deviceInput = deviceMemory.get();
    float* const deviceOutput = deviceInput + elementCount;

    CheckCudaError(cudaMemcpy(deviceInput, input.data(), bytes, cudaMemcpyHostToDevice),
                   "Failed to copy input to GPU");

    DispatchSoftmax(deviceInput, deviceOutput, rowCount, rowSize);
    CheckCudaError(cudaGetLastError(), "Failed to launch softmax kernel");

    std::vector<float> result(elementCount);
    CheckCudaError(cudaMemcpy(result.data(), deviceOutput, bytes, cudaMemcpyDeviceToHost),
                   "Failed to copy softmax result from GPU");

    return result;
}
