#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>

__global__ void naive_gemm_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {

        const float* a_row = A + row * n;
        float sum = 0.0f;

        int k = 0;

        for (; k + 3 < n; k += 4) {
            sum += __ldg(a_row + k)     * __ldg(B + k * n + col);
            sum += __ldg(a_row + k + 1) * __ldg(B + (k + 1) * n + col);
            sum += __ldg(a_row + k + 2) * __ldg(B + (k + 2) * n + col);
            sum += __ldg(a_row + k + 3) * __ldg(B + (k + 3) * n + col);
        }

        for (; k < n; ++k) {
            sum += __ldg(a_row + k) * __ldg(B + k * n + col);
        }

        C[row * n + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    const size_t bytes = static_cast<size_t>(n) * n * sizeof(float);
    std::vector<float> c(static_cast<size_t>(n) * n);

    static float* d_A = nullptr;
    static float* d_B = nullptr;
    static float* d_C = nullptr;
    static int cached_n = 0;

    if (cached_n != n) {
        if (d_A) cudaFree(d_A);
        if (d_B) cudaFree(d_B);
        if (d_C) cudaFree(d_C);
        
        cudaMalloc(&d_A, bytes);
        cudaMalloc(&d_B, bytes);
        cudaMalloc(&d_C, bytes);
        cached_n = n;
    }

    cudaMemcpy(d_A, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 8);
    dim3 blocksPerGrid((n + 31) / 32, (n + 7) / 8);

    naive_gemm_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);

    cudaMemcpy(c.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    return c;
}