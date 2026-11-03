#include "gemm_cublas.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace {

cublasHandle_t handle = nullptr;
float* d_a = nullptr;
float* d_b = nullptr;
float* d_c = nullptr;
size_t capacity = 0;

void Reserve(size_t elems) {
    if (!handle) cublasCreate(&handle);
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

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (n <= 0) return {};

    const size_t elems = static_cast<size_t>(n) * n;
    const size_t bytes = elems * sizeof(float);

    Reserve(elems);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha,
                d_b, n,
                d_a, n,
                &beta,
                d_c, n);

    std::vector<float> c(elems);

    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    return c;
}
