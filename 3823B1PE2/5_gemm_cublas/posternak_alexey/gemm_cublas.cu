#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdlib>

static float* d_a = nullptr;
static float* d_b = nullptr;
static float* d_c = nullptr;
static size_t allocated_n = 0;
static cublasHandle_t cublas_handle = nullptr;

static void cleanup_cuda_resources() {
    if (d_a) cudaFree(d_a);
    if (d_b) cudaFree(d_b);
    if (d_c) cudaFree(d_c);
    if (cublas_handle) cublasDestroy(cublas_handle);
    d_a = nullptr;
    d_b = nullptr;
    d_c = nullptr;
    cublas_handle = nullptr;
    allocated_n = 0;
}

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (n == 0) return std::vector<float>();
    
    size_t size = static_cast<size_t>(n) * n;
    
    static bool initialized = false;
    if (!initialized) {
        cublasCreate(&cublas_handle);
        std::atexit(cleanup_cuda_resources);
        initialized = true;
    }
    
    if (n > allocated_n) {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
        
        cudaMalloc(&d_a, size * sizeof(float));
        cudaMalloc(&d_b, size * sizeof(float));
        cudaMalloc(&d_c, size * sizeof(float));
        allocated_n = n;
    }
    
    cudaMemcpyAsync(d_a, a.data(), size * sizeof(float), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_b, b.data(), size * sizeof(float), cudaMemcpyHostToDevice, 0);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // C (m*n) = A (m*k) × B (k*n)
    // C^T (n*n) = B^T (n*n) × A^T (n*n)
    cublasSgemm(cublas_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha,
                d_b, n,
                d_a, n,
                &beta,
                d_c, n);
    
    std::vector<float> c(size);
    
    cudaMemcpyAsync(c.data(), d_c, size * sizeof(float), cudaMemcpyDeviceToHost, 0);
    
    cudaDeviceSynchronize();
    
    return c;
}
