#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    const size_t elements = static_cast<size_t>(n) * n;
    const size_t bytes = elements * sizeof(float);
    
    std::vector<float> c(elements);

    static float* d_A = nullptr;
    static float* d_B = nullptr;
    static float* d_C = nullptr;
    static cublasHandle_t handle = nullptr;
    static cudaStream_t stream = nullptr;
    static int cached_n = 0;

    if (cached_n != n) {
        if (d_A) {
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
        }
        if (handle) cublasDestroy(handle);
        if (stream) cudaStreamDestroy(stream);

        cudaMalloc(&d_A, bytes);
        cudaMalloc(&d_B, bytes);
        cudaMalloc(&d_C, bytes);
        
        cublasCreate(&handle);
        cudaStreamCreate(&stream);
        
        cached_n = n;
    }

    cudaMemcpyAsync(d_A, a.data(), bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, b.data(), bytes, cudaMemcpyHostToDevice, stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSetStream(handle, stream);

    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha, 
                d_B, n,
                d_A, n,
                &beta, 
                d_C, n);

    cudaMemcpyAsync(c.data(), d_C, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    return c;
}