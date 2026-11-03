#include "block_gemm_cuda.h"

#include <cuda_runtime.h>

namespace {

constexpr int kTileSize = 32;

__global__ void BlockGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                int n) {
    __shared__ float tile_a[kTileSize][kTileSize];
    __shared__ float tile_b[kTileSize][kTileSize];

    const int row = blockIdx.y * kTileSize + threadIdx.y;
    const int col = blockIdx.x * kTileSize + threadIdx.x;

    float sum = 0.0f;

    const int block_count = n / kTileSize;

    for (int tile = 0; tile < block_count; ++tile) {
        const int a_col = tile * kTileSize + threadIdx.x;
        const int b_row = tile * kTileSize + threadIdx.y;

        tile_a[threadIdx.y][threadIdx.x] = a[row * n + a_col];
        tile_b[threadIdx.y][threadIdx.x] = b[b_row * n + col];

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < kTileSize; ++k) {
            sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }

        __syncthreads();
    }

    c[row * n + col] = sum;
}

} // namespace

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    std::vector<float> c(static_cast<size_t>(n) * n);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    const size_t bytes = static_cast<size_t>(n) * n * sizeof(float);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(kTileSize, kTileSize);
    dim3 grid(n / kTileSize, n / kTileSize);

    BlockGemmKernel<<<grid, block>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}
