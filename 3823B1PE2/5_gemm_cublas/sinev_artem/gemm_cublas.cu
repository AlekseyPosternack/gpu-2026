#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <string>

inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

inline void checkCublas(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + " (cuBLAS error code: " + std::to_string(status) + ")");
    }
}

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    const size_t numElements = static_cast<size_t>(n) * n;
    const size_t bytes = numElements * sizeof(float);

    std::vector<float> c(numElements);

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    
    checkCuda(cudaMalloc(&d_a, bytes), "cudaMalloc d_a");
    checkCuda(cudaMalloc(&d_b, bytes), "cudaMalloc d_b");
    checkCuda(cudaMalloc(&d_c, bytes), "cudaMalloc d_c");

    checkCuda(cudaMemcpyAsync(d_a, a.data(), bytes, cudaMemcpyHostToDevice), "Memcpy H2D a");
    checkCuda(cudaMemcpyAsync(d_b, b.data(), bytes, cudaMemcpyHostToDevice), "Memcpy H2D b");

    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle), "cublasCreate");

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    checkCublas(cublasSgemm(handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            n, n, n,
                            &alpha,
                            d_b, n, 
                            d_a, n, 
                            &beta,
                            d_c, n), 
                "cublasSgemm");

    checkCuda(cudaMemcpyAsync(c.data(), d_c, bytes, cudaMemcpyDeviceToHost), "Memcpy D2H c");

    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    checkCublas(cublasDestroy(handle), "cublasDestroy");

    checkCuda(cudaFree(d_a), "cudaFree d_a");
    checkCuda(cudaFree(d_b), "cudaFree d_b");
    checkCuda(cudaFree(d_c), "cudaFree d_c");

    return c;
}