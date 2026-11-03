#include "naive_gemm_cuda.h"
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

__global__ void gemm_kernel(const float* __restrict__ a,
                            const float* __restrict__ b,
                            float* __restrict__ c, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    float s = 0.0f;
    #pragma unroll 4
    for (int k = 0; k < n; ++k) {
        s += a[i * n + k] * b[k * n + j];
    }
    c[i * n + j] = s;
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
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

    dim3 block(16, 16);
    dim3 grid((n + 15) / 16, (n + 15) / 16);
    gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
    CHECK_ERROR(cudaDeviceSynchronize());

    CHECK_ERROR(cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost));

    CHECK_ERROR(cudaFree(d_a));
    CHECK_ERROR(cudaFree(d_b));
    CHECK_ERROR(cudaFree(d_c));

    return c;
}