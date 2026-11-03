#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <cstddef>

namespace {
float *g_d_a = nullptr, *g_d_b = nullptr, *g_d_c = nullptr;
std::size_t g_cap = 0;

void ensureBuffers(std::size_t bytes) {
    if (bytes <= g_cap) return;
    if (g_d_a) cudaFree(g_d_a);
    if (g_d_b) cudaFree(g_d_b);
    if (g_d_c) cudaFree(g_d_c);
    cudaMalloc(&g_d_a, bytes);
    cudaMalloc(&g_d_b, bytes);
    cudaMalloc(&g_d_c, bytes);
    g_cap = bytes;
}

__global__ void NaiveGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                int n) {
    const int col = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const int row = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
    if (row >= n || col >= n) return;

    float4 acc0 = make_float4(0.f, 0.f, 0.f, 0.f); // row+0
    float4 acc1 = make_float4(0.f, 0.f, 0.f, 0.f); // row+1
    float4 acc2 = make_float4(0.f, 0.f, 0.f, 0.f); // row+2
    float4 acc3 = make_float4(0.f, 0.f, 0.f, 0.f); // row+3

    const float* a0 = a + (std::size_t)(row + 0) * n;
    const float* a1 = a + (std::size_t)(row + 1) * n;
    const float* a2 = a + (std::size_t)(row + 2) * n;
    const float* a3 = a + (std::size_t)(row + 3) * n;

    #pragma unroll 2
    for (int k = 0; k < n; ++k) {
        const float4 bv = *reinterpret_cast<const float4*>(
            b + (std::size_t)k * n + col);
        const float av0 = a0[k];
        const float av1 = a1[k];
        const float av2 = a2[k];
        const float av3 = a3[k];

        acc0.x += av0 * bv.x; acc0.y += av0 * bv.y; acc0.z += av0 * bv.z; acc0.w += av0 * bv.w;
        acc1.x += av1 * bv.x; acc1.y += av1 * bv.y; acc1.z += av1 * bv.z; acc1.w += av1 * bv.w;
        acc2.x += av2 * bv.x; acc2.y += av2 * bv.y; acc2.z += av2 * bv.z; acc2.w += av2 * bv.w;
        acc3.x += av3 * bv.x; acc3.y += av3 * bv.y; acc3.z += av3 * bv.z; acc3.w += av3 * bv.w;
    }

    std::size_t crow = (std::size_t)row * n + col;
    *reinterpret_cast<float4*>(c + crow + 0 * n) = acc0;
    *reinterpret_cast<float4*>(c + crow + 1 * n) = acc1;
    *reinterpret_cast<float4*>(c + crow + 2 * n) = acc2;
    *reinterpret_cast<float4*>(c + crow + 3 * n) = acc3;
}
} // namespace

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    std::size_t total = (std::size_t)n * n;
    std::vector<float> c(total, 0.0f);
    if (total == 0) return c;
    std::size_t bytes = total * sizeof(float);

    ensureBuffers(bytes);
    cudaMemcpy(g_d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(g_d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    const dim3 block(16, 16);
    const dim3 grid((n / 4 + block.x - 1) / block.x,
                    (n / 4 + block.y - 1) / block.y);
    NaiveGemmKernel<<<grid, block>>>(g_d_a, g_d_b, g_d_c, n);

    cudaMemcpy(c.data(), g_d_c, bytes, cudaMemcpyDeviceToHost);
    return c;
}