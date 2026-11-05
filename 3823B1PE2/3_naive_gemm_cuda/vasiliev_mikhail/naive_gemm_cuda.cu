#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace {

__global__ void GemmVec4Kernel(const float* __restrict__ a,
                               const float* __restrict__ b,
                               float* __restrict__ c, int n) {
    const int col = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n || col >= n) return;

    const float* a_row = a + static_cast<size_t>(row) * n;
    const float* b_col = b + col;

    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);

#pragma unroll 8
    for (int k = 0; k < n; ++k) {
        const float av = a_row[k];
        const float4 bv = *reinterpret_cast<const float4*>(
            b_col + static_cast<size_t>(k) * n);
        acc.x += av * bv.x;
        acc.y += av * bv.y;
        acc.z += av * bv.z;
        acc.w += av * bv.w;
    }

    *reinterpret_cast<float4*>(c + static_cast<size_t>(row) * n + col) = acc;
}

__global__ void GemmScalarKernel(const float* __restrict__ a,
                                 const float* __restrict__ b,
                                 float* __restrict__ c, int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n || col >= n) return;
    float acc = 0.f;
    for (int k = 0; k < n; ++k)
        acc += a[static_cast<size_t>(row) * n + k] *
               b[static_cast<size_t>(k) * n + col];
    c[static_cast<size_t>(row) * n + col] = acc;
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

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) return {};

    const size_t elems = static_cast<size_t>(n) * n;
    const size_t bytes = elems * sizeof(float);

    Reserve(elems);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    if (n % 4 == 0) {
        const dim3 block(16, 16);
        const dim3 grid((n / 4 + block.x - 1) / block.x,
                        (n + block.y - 1) / block.y);
        GemmVec4Kernel<<<grid, block>>>(d_a, d_b, d_c, n);
    } else {
        const dim3 block(16, 16);
        const dim3 grid((n + block.x - 1) / block.x,
                        (n + block.y - 1) / block.y);
        GemmScalarKernel<<<grid, block>>>(d_a, d_b, d_c, n);
    }

    std::vector<float> c(elems);

    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    return c;
}
