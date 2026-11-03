#include "block_gemm_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace {

constexpr int kTile = 32;
constexpr int kRows = 8;
constexpr int kPer = kTile / kRows;

__global__ void BlockGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c, int n) {
    __shared__ float As[kTile][kTile];
    __shared__ float Bs[kTile][kTile];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row_base = blockIdx.y * kTile;
    const int col = blockIdx.x * kTile + tx;

    float acc[kPer];
#pragma unroll
    for (int i = 0; i < kPer; ++i) acc[i] = 0.f;

    for (int kk = 0; kk < n; kk += kTile) {
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
            const int r = ty + i * kRows;
            const int ar = row_base + r;
            const int ac = kk + tx;
            As[r][tx] = (ar < n && ac < n) ? a[static_cast<size_t>(ar) * n + ac] : 0.f;
            const int br = kk + r;
            Bs[r][tx] = (br < n && col < n) ? b[static_cast<size_t>(br) * n + col] : 0.f;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < kTile; ++k) {
            const float bv = Bs[k][tx];
#pragma unroll
            for (int i = 0; i < kPer; ++i)
                acc[i] += As[ty + i * kRows][k] * bv;
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int row = row_base + ty + i * kRows;
        if (row < n && col < n) c[static_cast<size_t>(row) * n + col] = acc[i];
    }
}

float* d_a = nullptr;
float* d_b = nullptr;
float* d_c = nullptr;
size_t capacity = 0;

void Reserve(size_t elems) {
    if (elems <= capacity) return;
    cudaDeviceSynchronize();
    if (d_a) cudaFree(d_a);
    if (d_b) cudaFree(d_b);
    if (d_c) cudaFree(d_c);
    cudaMalloc(&d_a, elems * sizeof(float));
    cudaMalloc(&d_b, elems * sizeof(float));
    cudaMalloc(&d_c, elems * sizeof(float));
    capacity = elems;
}

}  // namespace

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) return {};

    const size_t elems = static_cast<size_t>(n) * n;
    const size_t bytes = elems * sizeof(float);

    Reserve(elems);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    const dim3 block(kTile, kRows);
    const dim3 grid((n + kTile - 1) / kTile, (n + kTile - 1) / kTile);
    BlockGemmKernel<<<grid, block>>>(d_a, d_b, d_c, n);

    std::vector<float> c(elems);

    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    return c;
}
