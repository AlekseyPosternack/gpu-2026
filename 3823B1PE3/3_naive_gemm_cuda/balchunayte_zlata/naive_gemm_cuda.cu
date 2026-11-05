#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <cstddef>

#define CUDA_CHECK(call) do {                              \
    cudaError_t err = (call);                              \
    if (err != cudaSuccess)                                \
        throw std::runtime_error(cudaGetErrorString(err)); \
} while (0)

// CUDA-ядро: один поток вычисляет один элемент C
__global__ void naiveGemmKernel(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        float sum = 0.0f;

        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }

        c[row * n + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(
    const std::vector<float>& a,
    const std::vector<float>& b,
    int n
) {
    if (n <= 0 ||
        static_cast<size_t>(n) * n != a.size() ||
        a.size() != b.size()) {
        throw std::invalid_argument("Invalid matrix size");
    }

    size_t bytes = a.size() * sizeof(float);
    std::vector<float> result(a.size());

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    try {
        CUDA_CHECK(cudaMalloc(&d_a, bytes));
        CUDA_CHECK(cudaMalloc(&d_b, bytes));
        CUDA_CHECK(cudaMalloc(&d_c, bytes));

        CUDA_CHECK(cudaMemcpy(
            d_a, a.data(), bytes, cudaMemcpyHostToDevice
        ));

        CUDA_CHECK(cudaMemcpy(
            d_b, b.data(), bytes, cudaMemcpyHostToDevice
        ));

        dim3 threads(16, 16);

        dim3 blocks(
            (n + threads.x - 1) / threads.x,
            (n + threads.y - 1) / threads.y
        );

        naiveGemmKernel<<<blocks, threads>>>(
            d_a, d_b, d_c, n
        );

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(
            result.data(), d_c, bytes,
            cudaMemcpyDeviceToHost
        ));

    } catch (...) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        throw;
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return result;
}
