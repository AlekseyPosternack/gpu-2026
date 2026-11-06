#include "gemm_cublas.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <thread>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                        \
            std::fprintf(stderr, "CUDA error \"%s\" at %s:%d\n",          \
                         cudaGetErrorString(err_), __FILE__, __LINE__);   \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

#define CUBLAS_CHECK(call)                                                \
    do {                                                                  \
        cublasStatus_t st_ = (call);                                      \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                               \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n",            \
                         static_cast<int>(st_), __FILE__, __LINE__);      \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

namespace {

struct CublasHandle {
    cublasHandle_t handle = nullptr;

    cublasHandle_t Get() {
        if (handle == nullptr) {
            CUBLAS_CHECK(cublasCreate(&handle));
        }
        return handle;
    }

    ~CublasHandle() {
        if (handle != nullptr) {
            cublasDestroy(handle);
        }
    }
};

struct DeviceBuffer {
    float* ptr = nullptr;
    size_t capacity = 0;

    float* Get(size_t count) {
        if (count > capacity) {
            if (ptr != nullptr) {
                CUDA_CHECK(cudaFree(ptr));
            }
            CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(float)));
            capacity = count;
        }
        return ptr;
    }

    ~DeviceBuffer() {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }
};

CublasHandle g_handle;
DeviceBuffer g_a, g_b, g_c;

}  // namespace

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (n <= 0) {
        return {};
    }
    const size_t count = static_cast<size_t>(n) * n;
    const size_t bytes = count * sizeof(float);

    cublasHandle_t handle = g_handle.Get();
    float* d_a = g_a.Get(count);
    float* d_b = g_b.Get(count);
    float* d_c = g_c.Get(count);

    std::vector<float> c;
    std::thread alloc_thread([&c, count] { c.resize(count); });

    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             n, n, n,
                             &alpha,
                             d_b, n,
                             d_a, n,
                             &beta,
                             d_c, n));

    alloc_thread.join();

    CUDA_CHECK(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    return c;
}