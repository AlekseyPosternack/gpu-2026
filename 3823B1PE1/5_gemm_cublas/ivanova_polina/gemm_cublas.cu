#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <mutex>

static float* d_a_cache = nullptr;
static float* d_b_cache = nullptr;
static float* d_c_cache = nullptr;
static size_t cached_bytes = 0;
static cublasHandle_t handle = nullptr;
static cudaStream_t stream = nullptr;
static bool initialized = false;
static std::mutex mtx;

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    int size = n * n;
    size_t bytes = size * sizeof(float);
    std::vector<float> c(size, 0.0f);

    if (n == 0) return c;

    std::lock_guard<std::mutex> lock(mtx);
    if (!initialized) {
        cublasCreate(&handle);
        cudaStreamCreate(&stream);
        cublasSetStream(handle, stream);
        
        cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
        
        initialized = true;
    }

    if (bytes > cached_bytes) {
        if (d_a_cache) cudaFree(d_a_cache);
        if (d_b_cache) cudaFree(d_b_cache);
        if (d_c_cache) cudaFree(d_c_cache);

        cudaMalloc(&d_a_cache, bytes);
        cudaMalloc(&d_b_cache, bytes);
        cudaMalloc(&d_c_cache, bytes);
        cached_bytes = bytes;
    }

    cudaMemcpyAsync(d_a_cache, a.data(), bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b_cache, b.data(), bytes, cudaMemcpyHostToDevice, stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha,
                d_b_cache, n,
                d_a_cache, n,
                &beta,
                d_c_cache, n);

    cudaMemcpyAsync(c.data(), d_c_cache, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    return c;
}