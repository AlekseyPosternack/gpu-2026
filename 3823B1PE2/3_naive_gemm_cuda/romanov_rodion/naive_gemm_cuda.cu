#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

__global__ void naive_gemm_kernel(const float* a, const float* b, float* c, int n) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    size_t size_bytes = n * n * sizeof(float);
    std::vector<float> c(n * n);

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;

    cudaMalloc(&d_a, size_bytes);
    cudaMalloc(&d_b, size_bytes);
    cudaMalloc(&d_c, size_bytes);

    cudaMemcpy(d_a, a.data(), size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size_bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    naive_gemm_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    cudaMemcpy(c.data(), d_c, size_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}