#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <mutex>

#define CHECK_ERROR(X)                            \
    do {                                          \
        cudaError_t err = (X);                    \
        if (err != cudaSuccess) {                 \
            printf("Error: %s\n",                 \
                   cudaGetErrorString(err));      \
            exit(0);                              \
        }                                         \
    } while (0)

#define CHECK_CUBLAS(X)                           \
    do {                                          \
        cublasStatus_t st = (X);                  \
        if (st != CUBLAS_STATUS_SUCCESS) {        \
            printf("cuBLAS error: %d\n", (int)st);\
            exit(0);                              \
        }                                         \
    } while (0)

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    std::vector<float> c(n * n);
    if (n == 0) return c;

    const size_t size = (size_t)n * n * sizeof(float);

    static cublasHandle_t handle;
    static std::once_flag handle_flag;
    std::call_once(handle_flag, []() {
        CHECK_CUBLAS(cublasCreate(&handle));
    });

    static float* d_a = nullptr;
    static float* d_b = nullptr;
    static float* d_c = nullptr;
    static size_t c_bytes = 0;

    if (size > c_bytes) {
        if (d_a) CHECK_ERROR(cudaFree(d_a));
        if (d_b) CHECK_ERROR(cudaFree(d_b));
        if (d_c) CHECK_ERROR(cudaFree(d_c));
        CHECK_ERROR(cudaMalloc(&d_a, size));
        CHECK_ERROR(cudaMalloc(&d_b, size));
        CHECK_ERROR(cudaMalloc(&d_c, size));
        c_bytes = size;
    }

    CHECK_ERROR(cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             n, n, n,
                             &alpha,
                             d_b, n,
                             d_a, n,
                             &beta,
                             d_c, n));

    CHECK_ERROR(cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost));

    return c;
}