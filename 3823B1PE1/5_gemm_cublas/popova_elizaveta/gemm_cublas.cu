#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>

namespace
{
    cublasHandle_t g_handle = nullptr;
    float *g_d_a = nullptr, *g_d_b = nullptr, *g_d_c = nullptr;
    size_t g_cap = 0;

    struct Cleanup
    {
        ~Cleanup()
        {
            if (g_handle)
                cublasDestroy(g_handle);
            if (g_d_a)
                cudaFree(g_d_a);
            if (g_d_b)
                cudaFree(g_d_b);
            if (g_d_c)
                cudaFree(g_d_c);
        }
    } g_cleanup;

    bool ensure(size_t elems)
    {
        if (!g_handle)
        {
            cudaSetDevice(0);
            cudaFree(0);
            if (cublasCreate(&g_handle) != CUBLAS_STATUS_SUCCESS)
                return false;
        }
        if (elems <= g_cap)
            return true;
        if (g_d_a)
        {
            cudaFree(g_d_a);
            cudaFree(g_d_b);
            cudaFree(g_d_c);
        }
        size_t bytes = elems * sizeof(float);
        if (cudaMalloc(&g_d_a, bytes) || cudaMalloc(&g_d_b, bytes) || cudaMalloc(&g_d_c, bytes))
            return false;
        g_cap = elems;
        return true;
    }
}

std::vector<float> GemmCUBLAS(const std::vector<float> &a,
                              const std::vector<float> &b, int n)
{
    std::vector<float> c(static_cast<size_t>(n) * n, 0.0f);
    if (n <= 0 || !ensure(c.size()))
        return c;

    size_t bytes = c.size() * sizeof(float);
    cudaMemcpy(g_d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(g_d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(g_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n, &alpha, g_d_b, n, g_d_a, n, &beta, g_d_c, n);

    cudaMemcpy(c.data(), g_d_c, bytes, cudaMemcpyDeviceToHost);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(err));
    return c;
}