#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <future>
#include <stdexcept>
#include <vector>

__global__ void GeluKernel(const float* input, float* output, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float x = input[i];
    float u = 0.7978845608f * (x + 0.044715f * x * x * x);
    output[i] = x / (1.0f + expf(-2.0f * u));
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t n = input.size();
    if (n == 0) return {};

    size_t bytes = n * sizeof(float);
    float* d_input = nullptr;
    float* d_output = nullptr;
    std::future<void> copy_future;

    auto check = [](cudaError_t error) {
        if (error != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(error));
        };

    auto cleanup = [&]() {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        };

    try {
        check(cudaMalloc((void**)&d_input, bytes));
        check(cudaMalloc((void**)&d_output, bytes));

        copy_future = std::async(std::launch::async, [&]() {
            check(cudaMemcpy(
                d_input, input.data(), bytes, cudaMemcpyHostToDevice
            ));
            check(cudaStreamSynchronize(0));
            });

        std::vector<float> result(n);
        copy_future.get();

        const int threads = 512;
        const int blocks = static_cast<int>((n + threads - 1) / threads);

        GeluKernel << <blocks, threads >> > (d_input, d_output, n);
        check(cudaGetLastError());

        check(cudaMemcpy(
            result.data(), d_output, bytes, cudaMemcpyDeviceToHost
        ));

        cleanup();
        return result;
    }
    catch (...) {
        if (copy_future.valid()) copy_future.wait();
        cleanup();
        throw;
    }
}
