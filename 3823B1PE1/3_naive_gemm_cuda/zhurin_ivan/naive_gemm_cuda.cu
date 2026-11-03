#include "naive_gemm_cuda.h"

#include <algorithm>
#include <cuda_runtime.h>

namespace {

constexpr int kBlockX = 32;
constexpr int kBlockY = 8;

__global__ void NaiveGemmKernel(const float* __restrict__ a,
                                 const float* __restrict__ b,
                                 float* __restrict__ c, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        const float* a_row = a + row * n;
        const float* b_col = b + col;

        float sum = 0.0f;
        int k = 0;
        #pragma unroll 4
        for (; k < n; ++k) {
            sum += a_row[k] * b_col[k * n];
        }
        c[row * n + col] = sum;
    }
}

}  // namespace

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                  const std::vector<float>& b,
                                  int n) {
    if (n <= 0) {
        return {};
    }
    const size_t count = static_cast<size_t>(n) * static_cast<size_t>(n);
    const size_t bytes = count * sizeof(float);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(kBlockX, kBlockY);
    dim3 grid((n + kBlockX - 1) / kBlockX, (n + kBlockY - 1) / kBlockY);
    NaiveGemmKernel<<<grid, block>>>(d_a, d_b, d_c, n);

    std::vector<float> result(count);
    cudaMemcpy(result.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return result;
}
