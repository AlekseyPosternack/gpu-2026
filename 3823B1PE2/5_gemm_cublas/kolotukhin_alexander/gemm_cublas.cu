#include "gemm_cublas.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(X)                                           \
    do {                                                        \
        cudaError_t error = (X);                                \
        if (error != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",       \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            std::exit(EXIT_FAILURE);                            \
        }                                                       \
    } while (0)

#define CHECK_CUBLAS(X)                                         \
    do {                                                        \
        cublasStatus_t status = (X);                            \
        if (status != CUBLAS_STATUS_SUCCESS) {                  \
            fprintf(stderr, "cuBLAS error at %s:%d — code %d\n",\
                    __FILE__, __LINE__, status);                \
            std::exit(EXIT_FAILURE);                            \
        }                                                       \
    } while (0)

namespace {

struct Cache {
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    
    float* h_a = nullptr;
    float* h_b = nullptr;
    float* h_c = nullptr;
    
    std::size_t capacity = 0;
    cublasHandle_t handle = nullptr;
    cudaStream_t stream = nullptr;

    ~Cache() {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
        if (h_a) cudaFreeHost(h_a);
        if (h_b) cudaFreeHost(h_b);
        if (h_c) cudaFreeHost(h_c);
        if (handle) cublasDestroy(handle);
        if (stream) cudaStreamDestroy(stream);
    }

    void init(std::size_t n) {
        if (handle == nullptr) {
            CHECK_CUDA(cudaSetDevice(0));
            CHECK_CUDA(cudaStreamCreate(&stream));
            CHECK_CUBLAS(cublasCreate(&handle));
            CHECK_CUBLAS(cublasSetStream(handle, stream));
        }
        if (n > capacity) {
            if (d_a) cudaFree(d_a);
            if (d_b) cudaFree(d_b);
            if (d_c) cudaFree(d_c);
            if (h_a) cudaFreeHost(h_a);
            if (h_b) cudaFreeHost(h_b);
            if (h_c) cudaFreeHost(h_c);
            
            const std::size_t bytes = n * n * sizeof(float);
            
            CHECK_CUDA(cudaMalloc(&d_a, bytes));
            CHECK_CUDA(cudaMalloc(&d_b, bytes));
            CHECK_CUDA(cudaMalloc(&d_c, bytes));
            
            CHECK_CUDA(cudaMallocHost(&h_a, bytes));
            CHECK_CUDA(cudaMallocHost(&h_b, bytes));
            CHECK_CUDA(cudaMallocHost(&h_c, bytes));
            
            capacity = n;
        }
    }
};

Cache& getCache() {
    static Cache c;
    return c;
}

} // namespace

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (n <= 0) {
        return {};
    }

    const std::size_t total_elements = static_cast<std::size_t>(n) * n;
    const std::size_t bytes = total_elements * sizeof(float);

    Cache& cache_ = getCache();
    cache_.init(static_cast<std::size_t>(n));

    std::memcpy(cache_.h_a, a.data(), bytes);
    std::memcpy(cache_.h_b, b.data(), bytes);

    CHECK_CUDA(cudaMemcpyAsync(cache_.d_a, cache_.h_a, bytes, cudaMemcpyHostToDevice, cache_.stream));
    CHECK_CUDA(cudaMemcpyAsync(cache_.d_b, cache_.h_b, bytes, cudaMemcpyHostToDevice, cache_.stream));

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    CHECK_CUBLAS(cublasSgemm(cache_.handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             n, n, n,
                             &alpha,
                             cache_.d_b, n,
                             cache_.d_a, n,
                             &beta,
                             cache_.d_c, n));

    std::vector<float> c(total_elements);

    CHECK_CUDA(cudaMemcpyAsync(cache_.h_c, cache_.d_c, bytes, cudaMemcpyDeviceToHost, cache_.stream));

    CHECK_CUDA(cudaStreamSynchronize(cache_.stream));

    std::memcpy(c.data(), cache_.h_c, bytes);

    return c;
}