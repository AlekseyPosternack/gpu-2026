#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

const int BLOCK_SIZE = 16;

__global__ void block_gemm_kernel(const float* __restrict__ a, 
                                  const float* __restrict__ b, 
                                  float* __restrict__ c, 
                                  int n) {
    __shared__ float s_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_b[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    float sum = 0.0f;

    int num_tiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int t = 0; t < num_tiles; ++t) {
        int col_a = t * BLOCK_SIZE + tx;
        int row_b = t * BLOCK_SIZE + ty;

        if (row < n && col_a < n) {
            s_a[ty][tx] = a[row * n + col_a];
        } else {
            s_a[ty][tx] = 0.0f;
        }

        if (row_b < n && col < n) {
            s_b[ty][tx] = b[row_b * n + col];
        } else {
            s_b[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += s_a[ty][k] * s_b[k][tx];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    size_t size_bytes = n * n * sizeof(float);
    std::vector<float> c(n * n, 0.0f);

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;

    cudaMalloc(&d_a, size_bytes);
    cudaMalloc(&d_b, size_bytes);
    cudaMalloc(&d_c, size_bytes);

    cudaMemcpy(d_a, a.data(), size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size_bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    block_gemm_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    cudaMemcpy(c.data(), d_c, size_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}