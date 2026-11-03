#include <cuda_runtime.h>
#include <cmath>

#include "gelu_cuda.h"

namespace {
// GELU(x) = x / (1 + exp(-x * (kA + kB * x^2)))
constexpr float kLog2e = 1.44269504089f;
constexpr float kA = 1.59576912f * kLog2e;
constexpr float kB = 0.0713548f * kLog2e;

constexpr int kBlockSize = 256;
}

__global__ void geluKernel(const float* __restrict__ input, float* __restrict__ output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float t = x * (kA + kB * x * x);
        output[idx] = x / (1.0f + exp2f(-t));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    if (input.empty()) {
        return {};
    }

    const size_t size = input.size();

    static float* d_in = nullptr;
    static float* d_out = nullptr;
    static size_t current_capacity = 0;

    if (current_capacity < size) {
        if (d_in) cudaFree(d_in);
        if (d_out) cudaFree(d_out);
        cudaMalloc(&d_in, size * sizeof(float));
        cudaMalloc(&d_out, size * sizeof(float));
        current_capacity = size;
    }

    cudaMemcpyAsync(d_in, input.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    int numBlocks = (size + kBlockSize - 1) / kBlockSize;

    geluKernel<<<numBlocks, kBlockSize>>>(d_in, d_out, size);

    std::vector<float> result(size);

    cudaMemcpyAsync(result.data(), d_out, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    return result;
}
