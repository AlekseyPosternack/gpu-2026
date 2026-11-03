#include "block_gemm_cuda.h"

#include <cuda_runtime.h>
#include <cstddef>

#define TILE_SIZE 32

__global__ void block_gemm_kernel(const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int n) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int m = 0; m < n / TILE_SIZE; ++m) {

        tile_A[ty][tx] = A[row * n + m * TILE_SIZE + tx];
        tile_B[ty][tx] = B[(m * TILE_SIZE + ty) * n + col];

        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_A[ty][k] * tile_B[k][tx];
        }

        __syncthreads();
    }

    C[row * n + col] = sum;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    const std::size_t total_elements = static_cast<std::size_t>(n) * n;
    std::vector<float> c(total_elements);

    if (n == 0) return c;

    const std::size_t bytes = total_elements * sizeof(float);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    if (cudaMalloc(&d_a, bytes) != cudaSuccess) return {};
    if (cudaMalloc(&d_b, bytes) != cudaSuccess) {
        cudaFree(d_a);
        return {};
    }
    if (cudaMalloc(&d_c, bytes) != cudaSuccess) {
        cudaFree(d_a); cudaFree(d_b);
        return {};
    }

    if (cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        return {};
    }
    if (cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        return {};
    }

    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize(n / TILE_SIZE, n / TILE_SIZE);

    block_gemm_kernel << <gridSize, blockSize >> > (d_a, d_b, d_c, n);

    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        return {};
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        return {};
    }

    if (cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        return {};
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}