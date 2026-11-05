#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <stdexcept>

#define CUDA_CHECK(call) do {                         \
    cudaError_t err = (call);                         \
    if (err != cudaSuccess)                           \
        throw std::runtime_error(cudaGetErrorString(err)); \
} while (0)

// Функция, выполняемая на GPU
__global__ void geluKernel(const float* input,
                           float* output,
                           size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float x = input[i];
        float x3 = x * x * x;

        float z = -1.5957691216f *
                  (x + 0.044715f * x3);

        output[i] = x / (1.0f + expf(z));
    }
}

// Функция, вызываемая с CPU
std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t n = input.size();
    std::vector<float> result(n);

    if (n == 0) {
        return result;
    }

    size_t bytes = n * sizeof(float);

    float* d_input = nullptr;
    float* d_output = nullptr;

    try {
        // Выделяем память на GPU
        CUDA_CHECK(cudaMalloc(&d_input, bytes));
        CUDA_CHECK(cudaMalloc(&d_output, bytes));

        // Копируем входные данные CPU -> GPU
        CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes,
                              cudaMemcpyHostToDevice));

        // Запускаем вычисления
        int threads = 256;
        int blocks = static_cast<int>((n + threads - 1) / threads);

        geluKernel<<<blocks, threads>>>(d_input, d_output, n);
        CUDA_CHECK(cudaGetLastError());

        // Копируем результат GPU -> CPU
        CUDA_CHECK(cudaMemcpy(result.data(), d_output, bytes,
                              cudaMemcpyDeviceToHost));

    } catch (...) {
        cudaFree(d_input);
        cudaFree(d_output);
        throw;
    }

    // Освобождаем память GPU
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return result;
}
