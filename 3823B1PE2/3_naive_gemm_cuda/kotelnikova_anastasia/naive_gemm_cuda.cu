#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>
#include <cstddef>

namespace {
constexpr int TILE = 32;

__global__ void GemmTiledKernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int n)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    const int numTiles = n / TILE;
    for (int t = 0; t < numTiles; ++t) {
        As[threadIdx.y][threadIdx.x] =
            A[row * n + (t * TILE + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] =
            B[(t * TILE + threadIdx.y) * n + col];

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    C[row * n + col] = sum;
}

float*      g_d_a = nullptr;
float*      g_d_b = nullptr;
float*      g_d_c = nullptr;
std::size_t g_cap = 0;

void ensureBuffers(std::size_t elems) {
    if (elems <= g_cap) return;
    if (g_d_a) cudaFree(g_d_a);
    if (g_d_b) cudaFree(g_d_b);
    if (g_d_c) cudaFree(g_d_c);
    const std::size_t bytes = elems * sizeof(float);
    cudaMalloc(&g_d_a, bytes);
    cudaMalloc(&g_d_b, bytes);
    cudaMalloc(&g_d_c, bytes);
    g_cap = elems;
}

} // namespace

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n)
{
    std::vector<float> c(static_cast<std::size_t>(n) * n, 0.0f);
    if (n <= 0) return c;

    const std::size_t elems = static_cast<std::size_t>(n) * n;
    const std::size_t bytes = elems * sizeof(float);

    ensureBuffers(elems);

    cudaMemcpyAsync(g_d_a, a.data(), bytes, cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(g_d_b, b.data(), bytes, cudaMemcpyHostToDevice, 0);

    dim3 block(TILE, TILE);
    dim3 grid(n / TILE, n / TILE);
    GemmTiledKernel<<<grid, block>>>(g_d_a, g_d_b, g_d_c, n);

    cudaMemcpyAsync(c.data(), g_d_c, bytes, cudaMemcpyDeviceToHost, 0);
    cudaDeviceSynchronize();

    return c;
}