#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
constexpr float kGeluCubic = 0.044715f;

__global__ void GeluKernel(const float* input, float* output, std::size_t size) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= size) {
        return;
    }

    const float x = input[index];
    const float x3 = x * x * x;
    const float z = kSqrtTwoOverPi * (x + kGeluCubic * x3);

    // tanh(z) = 2 / (1 + exp(-2z)) - 1.
    // __expf is considerably faster than tanhf and is sufficient for GELU.
    const float tanhApprox = 2.0f / (1.0f + __expf(-2.0f * z)) - 1.0f;
    output[index] = 0.5f * x * (1.0f + tanhApprox);
}

void CheckCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

} // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
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

        const std::size_t blocks =
            (input.size() + kThreadsPerBlock - 1) / kThreadsPerBlock;
        GeluKernel<<<static_cast<unsigned int>(blocks), kThreadsPerBlock>>>(
            deviceInput, deviceOutput, input.size());
        CheckCuda(cudaGetLastError(), "GeluKernel launch");
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