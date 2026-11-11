#include "gelu_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace {

void Check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

float* g_device = nullptr;
std::size_t g_capacity = 0;

float* GetDeviceBuffer(std::size_t size) {
    if (size > g_capacity) {
        if (g_device != nullptr) {
            cudaFree(g_device);
            g_device = nullptr;
        }
        Check(cudaMalloc(&g_device, size * sizeof(float)), "cudaMalloc");
        g_capacity = size;
    }
    return g_device;
}

__global__ void GeluKernel(float* data, std::size_t size) {
    const float k = 1.5957691216057308f;
    const float c = 0.044715f;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < size; i += stride) {
        const float x = data[i];
        const float t = k * x * (1.0f + c * x * x);
        data[i] = t < -80.0f ? 0.0f : __fdividef(x, 1.0f + __expf(-t));
    }
}

}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const std::size_t size = input.size();
    std::vector<float> output(size);
    if (size == 0) {
        return output;
    }

    const std::size_t bytes = size * sizeof(float);
    float* device = GetDeviceBuffer(size);

    Check(cudaMemcpy(device, input.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D");

    const int threads = 256;
    std::size_t blocks = (size + threads - 1) / threads;
    if (blocks > 65535 * 8) {
        blocks = 65535 * 8;
    }
    GeluKernel<<<static_cast<unsigned>(blocks), threads>>>(device, size);
    Check(cudaGetLastError(), "GeluKernel");

    Check(cudaMemcpy(output.data(), device, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    return output;
}
