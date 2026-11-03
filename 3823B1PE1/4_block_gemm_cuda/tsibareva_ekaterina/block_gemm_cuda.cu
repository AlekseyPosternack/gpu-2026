#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHECK_ERROR(X)                            \
    do {                                          \
        cudaError_t err = (X);                    \
        if (err != cudaSuccess) {                 \
            printf("Error: %s\n",                 \
                   cudaGetErrorString(err));      \
            exit(0);                              \
        }                                         \
    } while (0)

#define BLOCK_SIZE 32
#define TILE_SIZE 2

__global__ void block_gemm_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c, int n) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int blockCount = n / BLOCK_SIZE;
    int tid = ty * blockDim.x + tx;
    int step = blockDim.x * blockDim.y;

    // Justification: by increasing productivity, each thread computes TILE_SIZE x TILE_SIZE elements of C instead of one
    int row0 = blockIdx.y * BLOCK_SIZE + ty * TILE_SIZE;
    int col0 = blockIdx.x * BLOCK_SIZE + tx * TILE_SIZE;
    int li = ty * TILE_SIZE;
    int lj = tx * TILE_SIZE;

    float acc[TILE_SIZE][TILE_SIZE];
    #pragma unroll
    for (int p = 0; p < TILE_SIZE; ++p)
        #pragma unroll
        for (int q = 0; q < TILE_SIZE; ++q)
            acc[p][q] = 0.0f;

    for (int K = 0; K < blockCount; ++K) {
        for (int idx = tid; idx < BLOCK_SIZE * BLOCK_SIZE; idx += step) {
            int r = idx / BLOCK_SIZE;
            int cl = idx % BLOCK_SIZE;
            As[r][cl] = a[(blockIdx.y * BLOCK_SIZE + r) * n + K * BLOCK_SIZE + cl];
            Bs[r][cl] = b[(K * BLOCK_SIZE + r) * n + blockIdx.x * BLOCK_SIZE + cl];
        }

        __syncthreads();

        #pragma unroll
        for (int t = 0; t < BLOCK_SIZE; ++t) {
            float av[TILE_SIZE], bv[TILE_SIZE];
            #pragma unroll
            for (int p = 0; p < TILE_SIZE; ++p)
                av[p] = As[li + p][t];
            #pragma unroll
            for (int q = 0; q < TILE_SIZE; ++q)
                bv[q] = Bs[t][lj + q];
            #pragma unroll
            for (int p = 0; p < TILE_SIZE; ++p)
                #pragma unroll
                for (int q = 0; q < TILE_SIZE; ++q)
                    acc[p][q] += av[p] * bv[q];
        }

        __syncthreads();
    }

    #pragma unroll
    for (int p = 0; p < TILE_SIZE; ++p)
        #pragma unroll
        for (int q = 0; q < TILE_SIZE; ++q)
            c[(row0 + p) * n + col0 + q] = acc[p][q];
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    std::vector<float> c(n * n);
    if (n == 0) return c;

    const size_t size = (size_t)n * n * sizeof(float);

    float* d_a;
    float* d_b;
    float* d_c;
    CHECK_ERROR(cudaMalloc(&d_a, size));
    CHECK_ERROR(cudaMalloc(&d_b, size));
    CHECK_ERROR(cudaMalloc(&d_c, size));

    CHECK_ERROR(cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE / TILE_SIZE, BLOCK_SIZE / TILE_SIZE);
    dim3 grid(n / BLOCK_SIZE, n / BLOCK_SIZE);
    block_gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
    CHECK_ERROR(cudaDeviceSynchronize());

    CHECK_ERROR(cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost));

    CHECK_ERROR(cudaFree(d_a));
    CHECK_ERROR(cudaFree(d_b));
    CHECK_ERROR(cudaFree(d_c));

    return c;
}