#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <vector>

constexpr int TILE = 32;
constexpr int ROWS = 8;
constexpr int BLOCK_ROWS = TILE * 2;

__global__ void BlockGemmKernel(
    const float* a, const float* b, float* c, int n
) {
    __shared__ float tileA[BLOCK_ROWS][TILE];
    __shared__ float tileB[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = blockIdx.x * TILE + tx;
    int rowBase = blockIdx.y * BLOCK_ROWS;

    float sums[ROWS] = { 0.0f };

    for (int t = 0; t < (n + TILE - 1) / TILE; ++t) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            int localRow = ty + r * 8;
            int globalRow = rowBase + localRow;
            int k = t * TILE + tx;

            tileA[localRow][tx] =
                (globalRow < n && k < n)
                ? a[(size_t)globalRow * n + k] : 0.0f;
        }

#pragma unroll
        for (int r = 0; r < 4; ++r) {
            int localRow = ty + r * 8;
            int k = t * TILE + localRow;

            tileB[localRow][tx] =
                (k < n && col < n)
                ? b[(size_t)k * n + col] : 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
            float valueB = tileB[k][tx];

#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                sums[r] += tileA[ty + r * 8][k] * valueB;
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        int row = rowBase + ty + r * 8;

        if (row < n && col < n)
            c[(size_t)row * n + col] = sums[r];
    }
}

std::vector<float> BlockGemmCUDA(
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
        dim3 blocks(
            (n + TILE - 1) / TILE,
            (n + BLOCK_ROWS - 1) / BLOCK_ROWS
        );

        BlockGemmKernel << <blocks, threads >> > (d_a, d_b, d_c, n);
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