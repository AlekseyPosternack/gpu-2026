#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <future>
#include <stdexcept>

namespace {
void CheckCUDA(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

struct DeviceBuffer {
    float* data = nullptr;
    std::size_t capacity = 0;
    ~DeviceBuffer() { cudaFree(data); }
    void Reserve(std::size_t n) {
        if (n > capacity) {
            float* next = nullptr;
            CheckCUDA(cudaMalloc(&next, n * sizeof(float)));
            cudaFree(data);
            data = next;
            capacity = n;
        }
    }
};

__global__ void GeluKernel(float* data, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const float x = data[i];
        const float arg = x * (-1.5957691216057308f - 0.0713548162726009f * x * x);
        data[i] = __fdividef(x, 1.0f + __expf(arg));
    }
}
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    if (input.empty()) {
        return {};
    }
    
    const std::size_t n = input.size();
    const std::size_t bytes = n * sizeof(float);
    thread_local DeviceBuffer buffer;
    buffer.Reserve(n);
    auto output = std::async(std::launch::async, [n] { return std::vector<float>(n); });
    CheckCUDA(cudaMemcpy(buffer.data, input.data(), bytes, cudaMemcpyHostToDevice));
    const unsigned blocks = static_cast<unsigned>((n + 255) / 256);
    GeluKernel<<<blocks, 256>>>(buffer.data, n);
    CheckCUDA(cudaGetLastError());
    std::vector<float> result = output.get();
    CheckCUDA(cudaMemcpy(result.data(), buffer.data, bytes, cudaMemcpyDeviceToHost));
    return result;
}
