#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <vector>

namespace {
constexpr int TILE = 16;

__global__ void BlockGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;
    const int numTiles = n / TILE;

    for (int t = 0; t < numTiles; ++t) {
        As[ty][tx] = a[(std::size_t)row * n + (t * TILE + tx)];
        Bs[ty][tx] = b[(std::size_t)(t * TILE + ty) * n + col];

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            sum += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    c[(std::size_t)row * n + col] = sum;
}

float*      g_d_a = nullptr;
float*      g_d_b = nullptr;
float*      g_d_c = nullptr;
std::size_t g_cap = 0;

void ensureBuffers(std::size_t bytes) {
    if (bytes <= g_cap) return;
    if (g_d_a) cudaFree(g_d_a);
    if (g_d_b) cudaFree(g_d_b);
    if (g_d_c) cudaFree(g_d_c);
    cudaMalloc(&g_d_a, bytes);
    cudaMalloc(&g_d_b, bytes);
    cudaMalloc(&g_d_c, bytes);
    g_cap = bytes;
}
} // namespace

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    std::size_t total = (std::size_t)n * n;
    std::vector<float> c(total, 0.0f);
    if (total == 0) return c;
    std::size_t bytes = total * sizeof(float);

    ensureBuffers(bytes);
    cudaMemcpy(g_d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(g_d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 threads(TILE, TILE);
    dim3 blocks(n / TILE, n / TILE);
    BlockGemmKernel<<<blocks, threads>>>(g_d_a, g_d_b, g_d_c, n);

    cudaMemcpy(c.data(), g_d_c, bytes, cudaMemcpyDeviceToHost);
    return c;
}