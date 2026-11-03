#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <vector>

__global__ void GemmKernel(
    const float* a, const float* b, float* c, int n
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = (blockIdx.y * blockDim.y + threadIdx.y) * 8;

    if (row >= n || col >= n) return;

    float sums[8] = { 0.0f };

    for (int k = 0; k < n; ++k) {
        float valueB = b[(size_t)k * n + col];

#pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (row + r < n) {
                sums[r] += a[(size_t)(row + r) * n + k] * valueB;
            }
        }
    }

#pragma unroll
    for (int r = 0; r < 8; ++r) {
        if (row + r < n) {
            c[(size_t)(row + r) * n + col] = sums[r];
        }
    }
}

std::vector<float> NaiveGemmCUDA(
    const std::vector<float>& a,
    const std::vector<float>& b,
    int n
) {
    if (n <= 0)
        throw std::invalid_argument("Invalid n");

    size_t count = (size_t)n * n;

    if (a.size() != count || b.size() != count)
        throw std::invalid_argument("Invalid matrix size");

    size_t bytes = count * sizeof(float);
    std::vector<float> result(count);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    auto check = [](cudaError_t error) {
        if (error != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(error));
        };

    auto cleanup = [&]() {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
        };

    try {
        check(cudaMalloc((void**)&d_a, bytes));
        check(cudaMalloc((void**)&d_b, bytes));
        check(cudaMalloc((void**)&d_c, bytes));

        check(cudaMemcpy(
            d_a, a.data(), bytes, cudaMemcpyHostToDevice
        ));

        check(cudaMemcpy(
            d_b, b.data(), bytes, cudaMemcpyHostToDevice
        ));

        dim3 threads(32, 8);
        dim3 blocks((n + 31) / 32, (n + 63) / 64);

        GemmKernel << <blocks, threads >> > (d_a, d_b, d_c, n);
        check(cudaGetLastError());

        check(cudaMemcpy(
            result.data(), d_c, bytes, cudaMemcpyDeviceToHost
        ));
    }
    catch (...) {
        cleanup();
        throw;
    }

    cleanup();
    return result;
}