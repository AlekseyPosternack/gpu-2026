#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstddef>

namespace {
float *g_d_a = nullptr, *g_d_b = nullptr, *g_d_c = nullptr;
std::size_t g_cap = 0;

cublasHandle_t g_handle = nullptr;

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

void ensureHandle() {
    if (!g_handle) {
        cublasCreate(&g_handle);
    }
}
} // namespace

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    std::size_t total = (std::size_t)n * n;
    std::vector<float> c(total, 0.0f);
    if (total == 0) return c;

    std::size_t bytes = total * sizeof(float);

    ensureBuffers(bytes);
    ensureHandle();

    cudaMemcpy(g_d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(g_d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    
    cublasSgemm(g_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha,
                g_d_b, n,
                g_d_a, n,
                &beta,
                g_d_c, n);

    cudaMemcpy(c.data(), g_d_c, bytes, cudaMemcpyDeviceToHost);

    return c;
}