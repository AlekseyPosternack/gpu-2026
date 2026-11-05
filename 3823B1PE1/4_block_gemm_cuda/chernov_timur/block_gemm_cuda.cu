#include "block_gemm_cuda.h"
#include <cuda_runtime.h>

#define TILE 32

__global__ void block_gemm_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int n) {

    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE + ty;
    int col = bx * TILE + tx;

    float sum = 0.0f;
    int numTiles = n / TILE;

    for (int t = 0; t < numTiles; ++t) {

        As[ty][tx] = __ldg(&A[row * n + (t * TILE + tx)]);
        Bs[ty][tx] = __ldg(&B[(t * TILE + ty) * n + col]);

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    C[row * n + col] = sum;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    const size_t bytes = static_cast<size_t>(n) * n * sizeof(float);
    std::vector<float> c(static_cast<size_t>(n) * n);

    static float* d_A = nullptr;
    static float* d_B = nullptr;
    static float* d_C = nullptr;
    static int cached_n = 0;

    if (cached_n != n) {
        if (d_A) { cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); }
        cudaMalloc(&d_A, bytes);
        cudaMalloc(&d_B, bytes);
        cudaMalloc(&d_C, bytes);
        cached_n = n;
    }

    cudaMemcpy(d_A, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE, TILE);
    dim3 blocksPerGrid(n / TILE, n / TILE);

    block_gemm_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);

    cudaMemcpy(c.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    return c;
}
