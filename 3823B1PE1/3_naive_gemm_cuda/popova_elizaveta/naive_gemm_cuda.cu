#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstddef>

namespace
{

    __global__ void gemm_naive_kernel(const float *__restrict__ A,
                                      const float *__restrict__ B,
                                      float *__restrict__ C,
                                      int n)
    {
        const int row = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
        const int col = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
        if (row + 3 >= n || col + 3 >= n)
            return;

        float c00 = 0, c01 = 0, c02 = 0, c03 = 0;
        float c10 = 0, c11 = 0, c12 = 0, c13 = 0;
        float c20 = 0, c21 = 0, c22 = 0, c23 = 0;
        float c30 = 0, c31 = 0, c32 = 0, c33 = 0;

        for (int k = 0; k < n; ++k)
        {
            const float a0 = A[(row + 0) * n + k];
            const float a1 = A[(row + 1) * n + k];
            const float a2 = A[(row + 2) * n + k];
            const float a3 = A[(row + 3) * n + k];

            const float4 b = *reinterpret_cast<const float4 *>(B + k * n + col);

            c00 += a0 * b.x;
            c01 += a0 * b.y;
            c02 += a0 * b.z;
            c03 += a0 * b.w;
            c10 += a1 * b.x;
            c11 += a1 * b.y;
            c12 += a1 * b.z;
            c13 += a1 * b.w;
            c20 += a2 * b.x;
            c21 += a2 * b.y;
            c22 += a2 * b.z;
            c23 += a2 * b.w;
            c30 += a3 * b.x;
            c31 += a3 * b.y;
            c32 += a3 * b.z;
            c33 += a3 * b.w;
        }

        C[(row + 0) * n + col + 0] = c00;
        C[(row + 0) * n + col + 1] = c01;
        C[(row + 0) * n + col + 2] = c02;
        C[(row + 0) * n + col + 3] = c03;
        C[(row + 1) * n + col + 0] = c10;
        C[(row + 1) * n + col + 1] = c11;
        C[(row + 1) * n + col + 2] = c12;
        C[(row + 1) * n + col + 3] = c13;
        C[(row + 2) * n + col + 0] = c20;
        C[(row + 2) * n + col + 1] = c21;
        C[(row + 2) * n + col + 2] = c22;
        C[(row + 2) * n + col + 3] = c23;
        C[(row + 3) * n + col + 0] = c30;
        C[(row + 3) * n + col + 1] = c31;
        C[(row + 3) * n + col + 2] = c32;
        C[(row + 3) * n + col + 3] = c33;
    }

} // namespace

std::vector<float> NaiveGemmCUDA(const std::vector<float> &a,
                                 const std::vector<float> &b,
                                 int n)
{
    std::vector<float> c(static_cast<std::size_t>(n) * n, 0.0f);
    if (n == 0)
        return c;

    const std::size_t bytes = static_cast<std::size_t>(n) * n * sizeof(float);

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    constexpr int TX = 16;
    constexpr int TY = 16;
    dim3 block(TX, TY);
    dim3 grid((n + TX * 4 - 1) / (TX * 4), (n + TY * 4 - 1) / (TY * 4));

    gemm_naive_kernel<<<grid, block>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return c;
}