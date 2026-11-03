#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>


namespace {
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    size_t allocated_elements = 0;

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    constexpr size_t element_size = sizeof(float);
} // namespace

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    static cublasHandle_t handle = nullptr;
    if (!handle) {
        cublasCreate(&handle);
    }

    size_t num_elements = static_cast<size_t>(n) * n;
    size_t required_bytes = num_elements * element_size;

    if (num_elements > allocated_elements) {
        if (d_A) cudaFree(d_A);
        if (d_B) cudaFree(d_B);
        if (d_C) cudaFree(d_C);

        cudaMalloc(&d_A, required_bytes);
        cudaMalloc(&d_B, required_bytes);
        cudaMalloc(&d_C, required_bytes);

        allocated_elements = num_elements;
    }

    cudaMemcpy(d_A, a.data(), required_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b.data(), required_bytes, cudaMemcpyHostToDevice);

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha,
                d_B, n,
                d_A, n,
                &beta,
                d_C, n);

    std::vector<float> c(num_elements);
    cudaMemcpy(c.data(), d_C, required_bytes, cudaMemcpyDeviceToHost);
    return c;
}