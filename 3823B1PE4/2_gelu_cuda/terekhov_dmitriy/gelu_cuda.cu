#include "gelu_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>

namespace {

void checkCuda(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

__global__ void geluKernel(float* data, std::size_t size) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size) {
        const float x = data[i];
        const float exponent = 1.5957691216057308f * x * (1.0f + 0.044715f * x * x);
        data[i] = x / (1.0f + expf(-exponent));
    }
}

}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    if (input.empty()) {
        return {};
    }

    const std::size_t bytes = input.size() * sizeof(float);
    float* device_data = nullptr;
    checkCuda(cudaMalloc(&device_data, bytes));

    std::vector<float> result;
    try {
        checkCuda(cudaMemcpy(device_data, input.data(), bytes, cudaMemcpyHostToDevice));

        constexpr unsigned int threads = 256;
        const unsigned int blocks = static_cast<unsigned int>((input.size() + threads - 1) / threads);
        geluKernel<<<blocks, threads>>>(device_data, input.size());
        checkCuda(cudaGetLastError());

        result.resize(input.size());
        checkCuda(cudaMemcpy(result.data(), device_data, bytes, cudaMemcpyDeviceToHost));
    } catch (...) {
        cudaFree(device_data);
        throw;
    }

    checkCuda(cudaFree(device_data));
    return result;
}
