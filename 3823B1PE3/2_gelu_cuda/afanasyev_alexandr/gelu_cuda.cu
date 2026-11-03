#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace {

__device__ __forceinline__ float GeluElement(float x) {
    constexpr float kSqrtTwoOverPi = 0.7978845608f;
    const float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(kSqrtTwoOverPi * (x + 0.044715f * x3)));
}

__global__ void GeluKernel(const float* input, float* output, size_t count) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count) {
        output[idx] = GeluElement(input[idx]);
    }
}

}  // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    if (input.empty()) {
        return {};
    }

    const size_t count = input.size();

    float* d_input = nullptr;
    float* d_output = nullptr;

    cudaError_t err = cudaMalloc(&d_input, count * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc for input failed: ") +
                                 cudaGetErrorString(err));
    }

    err = cudaMalloc(&d_output, count * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_input);
        throw std::runtime_error(std::string("cudaMalloc for output failed: ") +
                                 cudaGetErrorString(err));
    }

    err = cudaMemcpy(d_input, input.data(), count * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        throw std::runtime_error(std::string("cudaMemcpy HostToDevice failed: ") +
                                 cudaGetErrorString(err));
    }

    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>((count + threads_per_block - 1) / threads_per_block);

    GeluKernel<<<blocks, threads_per_block>>>(d_input, d_output, count);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        throw std::runtime_error(std::string("GeluKernel launch failed: ") +
                                 cudaGetErrorString(err));
    }

    std::vector<float> output(count);
    err = cudaMemcpy(output.data(), d_output, count * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        throw std::runtime_error(std::string("cudaMemcpy DeviceToHost failed: ") +
                                 cudaGetErrorString(err));
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}
