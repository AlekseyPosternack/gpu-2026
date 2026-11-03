#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

namespace {

constexpr int kTile = 16;
constexpr int kReg = 4;

__global__ void NaiveGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                int n) {
    const int row = (blockIdx.y * kTile + threadIdx.y) * kReg;
    const int col = (blockIdx.x * kTile + threadIdx.x) * kReg;

    float acc[4][4] = {};

    for (int k = 0; k < n; ++k) {
        float a0 = a[(row + 0) * n + k];
        float a1 = a[(row + 1) * n + k];
        float a2 = a[(row + 2) * n + k];
        float a3 = a[(row + 3) * n + k];

        const float* b_row = b + k * n + col;

        float b0 = b_row[0];
        float b1 = b_row[1];
        float b2 = b_row[2];
        float b3 = b_row[3];

        acc[0][0] += a0 * b0;
        acc[0][1] += a0 * b1;
        acc[0][2] += a0 * b2;
        acc[0][3] += a0 * b3;

        acc[1][0] += a1 * b0;
        acc[1][1] += a1 * b1;
        acc[1][2] += a1 * b2;
        acc[1][3] += a1 * b3;

        acc[2][0] += a2 * b0;
        acc[2][1] += a2 * b1;
        acc[2][2] += a2 * b2;
        acc[2][3] += a2 * b3;

        acc[3][0] += a3 * b0;
        acc[3][1] += a3 * b1;
        acc[3][2] += a3 * b2;
        acc[3][3] += a3 * b3;
    }

    for (int i = 0; i < kReg; ++i) {
        for (int j = 0; j < kReg; ++j) {
            c[(row + i) * n + col + j] = acc[i][j];
        }
    }
}

} // namespace

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    std::vector<float> c(static_cast<size_t>(n) * n);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    const size_t bytes = static_cast<size_t>(n) * n * sizeof(float);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(kTile, kTile);
    dim3 grid(n / (kTile * kReg), n / (kTile * kReg));

    NaiveGemmKernel<<<grid, block>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}
