#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "gemm_cublas.h"

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (n == 0) {
        return {};
    }

    const size_t elements = static_cast<size_t>(n) * n;
    const size_t bytes = elements * sizeof(float);

    static cublasHandle_t handle = nullptr;
    if (!handle) {
        cublasCreate(&handle);
    }

    static float* d_a = nullptr;
    static float* d_b = nullptr;
    static float* d_c = nullptr;
    static size_t current_capacity = 0;

    if (elements > current_capacity) {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);
        current_capacity = elements;
    }

    cudaMemcpyAsync(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha,
                d_b, n,
                d_a, n,
                &beta,
                d_c, n);

    std::vector<float> result(elements);

    cudaMemcpyAsync(result.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    return result;
}
