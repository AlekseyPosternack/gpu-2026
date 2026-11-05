#include "softmax_cuda.h"

#include <cuda_runtime.h>

#include <cfloat>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

constexpr int kThreadsPerBlock = 256;

__device__ float BlockReduceMax(float value, float* shared) {
    const int lane = threadIdx.x;
    shared[lane] = value;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            shared[lane] = fmaxf(shared[lane], shared[lane + offset]);
        }
        __syncthreads();
    }
    return shared[0];
}

__device__ float BlockReduceSum(float value, float* shared) {
    const int lane = threadIdx.x;
    shared[lane] = value;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            shared[lane] += shared[lane + offset];
        }
        __syncthreads();
    }
    return shared[0];
}

__global__ void SoftmaxKernel(const float* __restrict__ input,
                              float* __restrict__ output,
                              int row_count,
                              int row_size) {
    const int row = blockIdx.x;
    if (row >= row_count) {
        return;
    }

    extern __shared__ float reduction[];
    const int thread = threadIdx.x;
    const int row_offset = row * row_size;

    float local_max = -FLT_MAX;
    for (int column = thread; column < row_size;
         column += blockDim.x) {
        local_max = fmaxf(local_max, input[row_offset + column]);
    }
    const float row_max = BlockReduceMax(local_max, reduction);

    float local_sum = 0.0f;
    for (int column = thread; column < row_size;
         column += blockDim.x) {
        local_sum += __expf(input[row_offset + column] - row_max);
    }
    const float row_sum = BlockReduceSum(local_sum, reduction);

    for (int column = thread; column < row_size;
         column += blockDim.x) {
        output[row_offset + column] =
            __expf(input[row_offset + column] - row_max) / row_sum;
    }
}

void CheckCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

} // namespace

std::vector<float> SoftmaxCUDA(const std::vector<float>& input,
                               int row_count) {
    if (row_count <= 0) {
        throw std::invalid_argument("row_count must be positive");
    }
    if (input.size() % static_cast<std::size_t>(row_count) != 0) {
        throw std::invalid_argument(
            "Input size must be divisible by row_count");
    }

    const std::size_t row_size_size =
        input.size() / static_cast<std::size_t>(row_count);
    if (row_size_size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("Row size is too large");
    }
    const int row_size = static_cast<int>(row_size_size);
    std::vector<float> result(input.size());
    if (input.empty()) {
        return result;
    }

    float* deviceInput = nullptr;
    float* deviceOutput = nullptr;
    const std::size_t bytes = input.size() * sizeof(float);

    try {
        CheckCuda(cudaMalloc(&deviceInput, bytes), "cudaMalloc(input)");
        CheckCuda(cudaMalloc(&deviceOutput, bytes), "cudaMalloc(output)");
        CheckCuda(cudaMemcpy(deviceInput, input.data(), bytes,
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(input)");

        SoftmaxKernel<<<row_count, kThreadsPerBlock,
                        kThreadsPerBlock * sizeof(float)>>>(
            deviceInput, deviceOutput, row_count, row_size);
        CheckCuda(cudaGetLastError(), "SoftmaxKernel launch");
        CheckCuda(cudaMemcpy(result.data(), deviceOutput, bytes,
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(output)");

        CheckCuda(cudaFree(deviceOutput), "cudaFree(output)");
        deviceOutput = nullptr;
        CheckCuda(cudaFree(deviceInput), "cudaFree(input)");
        deviceInput = nullptr;
    } catch (...) {
        if (deviceOutput != nullptr) {
            cudaFree(deviceOutput);
        }
        if (deviceInput != nullptr) {
            cudaFree(deviceInput);
        }
        throw;
    }

    return result;
}
