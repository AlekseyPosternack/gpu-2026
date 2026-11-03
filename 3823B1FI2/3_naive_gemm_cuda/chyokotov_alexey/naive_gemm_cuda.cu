#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>

constexpr int B = 16;

__global__ void gemm_kernel(const float* A, const float* Bm,
                            float* C, int n) {
    int row = blockIdx.y * B + threadIdx.y;
    int col = blockIdx.x * B + threadIdx.x;
    if (row >= n || col >= n) return;

    __shared__ float Bs[B][B + 1];
    float acc = 0.0f;

    for (int k0 = 0; k0 < n; k0 += B) {
        int bk = k0 + threadIdx.y;
        int bj = blockIdx.x * B + threadIdx.x;
        Bs[threadIdx.y][threadIdx.x] = (bk < n && bj < n) ? Bm[bk * n + bj] : 0.0f;
        __syncthreads();

        const float* a_row = A + row * n + k0;
        #pragma unroll
        for (int t = 0; t < B; ++t) {
            acc += a_row[t] * Bs[t][threadIdx.x];
        }
        __syncthreads();
    }

    C[row * n + col] = acc;
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    size_t bytes = (size_t)n * n * sizeof(float);
    std::vector<float> c((size_t)n * n);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(B, B);
    dim3 grid((n + B - 1) / B, (n + B - 1) / B);
    gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return c;
}