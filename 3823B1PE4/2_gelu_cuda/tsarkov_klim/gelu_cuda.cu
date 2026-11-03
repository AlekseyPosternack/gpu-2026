#include "gelu_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void CheckCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

struct DeviceBuffers {
    float* input = nullptr;
    float* output = nullptr;

    ~DeviceBuffers() {
        if (input != nullptr) {
            cudaFree(input);
        }
        if (output != nullptr) {
            cudaFree(output);
        }
    }
};

__global__ void GeluKernel(
    const float* input,
    float* output,
    std::size_t size) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < size) {
        const float x = input[i];
        const float x3 = x * x * x;
        const float inner = 0.7978845608f * (x + 0.044715f * x3);

        output[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

}  // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    std::vector<float> output(input.size());

    if (input.empty()) {
        return output;
    }

    DeviceBuffers device;

    const std::size_t bytes = input.size() * sizeof(float);

    CheckCuda(
        cudaMalloc(&device.input, bytes),
        "cudaMalloc input");

    CheckCuda(
        cudaMalloc(&device.output, bytes),
        "cudaMalloc output");

    CheckCuda(
        cudaMemcpy(
            device.input,
            input.data(),
            bytes,
            cudaMemcpyHostToDevice),
        "cudaMemcpy input to device");

    constexpr int threadsPerBlock = 256;
    const unsigned int blocks =
        static_cast<unsigned int>(
            (input.size() + threadsPerBlock - 1) / threadsPerBlock);

    GeluKernel<<<blocks, threadsPerBlock>>>(
        device.input,
        device.output,
        input.size());

    CheckCuda(cudaGetLastError(), "GeluKernel launch");

    CheckCuda(
        cudaMemcpy(
            output.data(),
            device.output,
            bytes,
            cudaMemcpyDeviceToHost),
        "cudaMemcpy output to host");

    return output;
}
