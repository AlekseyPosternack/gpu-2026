#include "gemm_cublas.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstddef>

namespace {

    cublasHandle_t g_handle = nullptr;
    float*      g_d_a = nullptr;
    float*      g_d_b = nullptr;
    float*      g_d_c = nullptr;
    std::size_t g_cap = 0;

    void ensureBuffers(std::size_t elems) {
        if (g_handle == nullptr) {
            cublasCreate(&g_handle);
        }
        if (elems <= g_cap) return;
        if (g_d_a) cudaFree(g_d_a);
        if (g_d_b) cudaFree(g_d_b);
        if (g_d_c) cudaFree(g_d_c);
        const std::size_t bytes = elems * sizeof(float);
        cudaMalloc(&g_d_a, bytes);
        cudaMalloc(&g_d_b, bytes);
        cudaMalloc(&g_d_c, bytes);
        g_cap = elems;
    }

} // namespace

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n)
{
    std::vector<float> c(static_cast<std::size_t>(n) * n, 0.0f);
    if (n <= 0) return c;

    const std::size_t elems = static_cast<std::size_t>(n) * n;
    const std::size_t bytes = elems * sizeof(float);

    ensureBuffers(elems);

    cudaMemcpyAsync(g_d_a, a.data(), bytes, cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(g_d_b, b.data(), bytes, cudaMemcpyHostToDevice, 0);

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
    cudaMemcpyAsync(c.data(), g_d_c, bytes, cudaMemcpyDeviceToHost, 0);
    cudaDeviceSynchronize();

    return c;
}