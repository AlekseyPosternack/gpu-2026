#include "block_gemm_cuda.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace {

constexpr int kTileSize = 16;

__global__ void BlockGemmKernel(const float* a,
                                const float* b,
                                float* c,
                                int n) {
    __shared__ float as[kTileSize][kTileSize];
    __shared__ float bs[kTileSize][kTileSize];

    const int row = blockIdx.y * kTileSize + threadIdx.y;
    const int col = blockIdx.x * kTileSize + threadIdx.x;

    float sum = 0.0f;

    for (int tile = 0; tile < (n + kTileSize - 1) / kTileSize; ++tile) {
        const int a_row = row;
        const int a_col = tile * kTileSize + threadIdx.x;
        const int b_row = tile * kTileSize + threadIdx.y;
        const int b_col = col;

        if (a_row < n && a_col < n) {
            as[threadIdx.y][threadIdx.x] = a[a_row * n + a_col];
        } else {
            as[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (b_row < n && b_col < n) {
            bs[threadIdx.y][threadIdx.x] = b[b_row * n + b_col];
        } else {
            bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < kTileSize; ++k) {
            sum += as[threadIdx.y][k] * bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

}  // namespace

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) {
        return {};
    }

    const size_t matrix_size = static_cast<size_t>(n) * static_cast<size_t>(n);
    if (a.size() != matrix_size || b.size() != matrix_size) {
        throw std::invalid_argument("A and B must both have size n * n");
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    cudaError_t err = cudaMalloc(&d_a, matrix_size * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc A failed: ") + cudaGetErrorString(err));
    }

    err = cudaMalloc(&d_b, matrix_size * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_a);
        throw std::runtime_error(std::string("cudaMalloc B failed: ") + cudaGetErrorString(err));
    }

    err = cudaMalloc(&d_c, matrix_size * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        throw std::runtime_error(std::string("cudaMalloc C failed: ") + cudaGetErrorString(err));
    }

    err = cudaMemcpy(d_a, a.data(), matrix_size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        throw std::runtime_error(std::string("cudaMemcpy A failed: ") + cudaGetErrorString(err));
    }

    err = cudaMemcpy(d_b, b.data(), matrix_size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        throw std::runtime_error(std::string("cudaMemcpy B failed: ") + cudaGetErrorString(err));
    }

    const int grid_x = (n + kTileSize - 1) / kTileSize;
    const int grid_y = (n + kTileSize - 1) / kTileSize;
    dim3 threads(kTileSize, kTileSize);
    dim3 grid(grid_x, grid_y);

    BlockGemmKernel<<<grid, threads>>>(d_a, d_b, d_c, n);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        throw std::runtime_error(std::string("BlockGemmKernel launch failed: ") + cudaGetErrorString(err));
    }

    std::vector<float> c(matrix_size);
    err = cudaMemcpy(c.data(), d_c, matrix_size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        throw std::runtime_error(std::string("cudaMemcpy C failed: ") + cudaGetErrorString(err));
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}
