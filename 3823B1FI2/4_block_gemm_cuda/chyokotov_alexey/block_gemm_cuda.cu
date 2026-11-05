#include "block_gemm_cuda.h"
#include <cuda_runtime.h>

constexpr int BLOCK = 32;

__global__ void block_gemm_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int n) {
    __shared__ float As[BLOCK][BLOCK];
    __shared__ float Bs[BLOCK][BLOCK];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * BLOCK + ty;
    int col = blockIdx.x * BLOCK + tx;
    float acc = 0.0f;

    for (int k0 = 0; k0 < n; k0 += BLOCK) {
        As[ty][tx] = A[row * n + k0 + tx];
        Bs[ty][tx] = B[(k0 + ty) * n + col];
        __syncthreads();

        #pragma unroll
        for (int t = 0; t < BLOCK; ++t) {
            acc += As[ty][t] * Bs[t][tx];
        }
        __syncthreads();
    }

    C[row * n + col] = acc;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    size_t bytes = (size_t)n * n * sizeof(float);
    std::vector<float> c((size_t)n * n);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(BLOCK, BLOCK);
    dim3 grid(n / BLOCK, n / BLOCK);
    block_gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return c;
}