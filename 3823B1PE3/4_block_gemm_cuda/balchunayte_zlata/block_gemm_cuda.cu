#include "block_gemm_cuda.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <cstddef>

#define CUDA_CHECK(call) do {                              \
    cudaError_t err = (call);                              \
    if (err != cudaSuccess)                                \
        throw std::runtime_error(cudaGetErrorString(err)); \
} while (0)

// Размер блока матрицы
constexpr int TILE = 16;

// CUDA-ядро
__global__ void blockGemmKernel(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    int numTiles = (n + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE + tx;
        int bRow = t * TILE + ty;

        // Загружаем блок A в shared memory
        if (row < n && aCol < n) {
            tileA[ty][tx] = a[row * n + aCol];
        } else {
            tileA[ty][tx] = 0.0f;
        }

        // Загружаем блок B в shared memory
        if (bRow < n && col < n) {
            tileB[ty][tx] = b[bRow * n + col];
        } else {
            tileB[ty][tx] = 0.0f;
        }

        // Ждём загрузки всех элементов
        __syncthreads();

        // Умножаем два блока
        for (int k = 0; k < TILE; ++k) {
            sum += tileA[ty][k] * tileB[k][tx];
        }

        // Ждём окончания вычислений перед
        // загрузкой следующей пары блоков
        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

// Функция, вызываемая с CPU
std::vector<float> BlockGemmCUDA(
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

        dim3 threads(TILE, TILE);

        dim3 blocks(
            (n + TILE - 1) / TILE,
            (n + TILE - 1) / TILE
        );

        blockGemmKernel<<<blocks, threads>>>(
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
