#include "gemm_cublas.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err__ = (call);                                        \
        if (err__ != cudaSuccess) {                                        \
            throw std::runtime_error(std::string("CUDA error: ") +         \
                                      cudaGetErrorString(err__));           \
        }                                                                  \
    } while (0)

#define CUBLAS_CHECK(call)                                                 \
    do {                                                                   \
        cublasStatus_t st__ = (call);                                      \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                               \
            throw std::runtime_error(                                      \
                std::string("cuBLAS error, status code: ") +               \
                std::to_string(static_cast<int>(st__)));                   \
        }                                                                  \
    } while (0)

namespace {

struct CublasResources {
    cublasHandle_t handle = nullptr;
    cudaStream_t   stream = nullptr;
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    size_t capacityElems = 0;
    bool ready = false;

    void EnsureReady(size_t elems) {
        if (!ready) {
            CUDA_CHECK(cudaStreamCreate(&stream));
            CUBLAS_CHECK(cublasCreate(&handle));
            CUBLAS_CHECK(cublasSetStream(handle, stream));
            cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
            ready = true;
        }
        if (elems > capacityElems) {
            if (d_a) CUDA_CHECK(cudaFree(d_a));
            if (d_b) CUDA_CHECK(cudaFree(d_b));
            if (d_c) CUDA_CHECK(cudaFree(d_c));
            CUDA_CHECK(cudaMalloc(&d_a, elems * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_b, elems * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_c, elems * sizeof(float)));
            capacityElems = elems;
        }
    }

    ~CublasResources() {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
        if (handle) cublasDestroy(handle);
        if (stream) cudaStreamDestroy(stream);
    }
};

CublasResources g_res;

}  // namespace

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    const size_t elems = (size_t)n * (size_t)n;
    std::vector<float> c(elems);
    if (n == 0) return c;

    g_res.EnsureReady(elems);
    cudaStream_t stream = g_res.stream;

    float* h_a = const_cast<float*>(a.data());
    float* h_b = const_cast<float*>(b.data());
    float* h_c = c.data();

    cudaError_t regA = cudaHostRegister(h_a, elems * sizeof(float), cudaHostRegisterDefault);
    cudaError_t regB = cudaHostRegister(h_b, elems * sizeof(float), cudaHostRegisterDefault);
    cudaError_t regC = cudaHostRegister(h_c, elems * sizeof(float), cudaHostRegisterDefault);

    CUDA_CHECK(cudaMemcpyAsync(g_res.d_a, h_a, elems * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(g_res.d_b, h_b, elems * sizeof(float),
                               cudaMemcpyHostToDevice, stream));

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    CUBLAS_CHECK(cublasSgemm(
        g_res.handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n,
        &alpha,
        g_res.d_b, n,   
        g_res.d_a, n,   
        &beta,
        g_res.d_c, n));

    CUDA_CHECK(cudaMemcpyAsync(h_c, g_res.d_c, elems * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (regA == cudaSuccess) cudaHostUnregister(h_a);
    if (regB == cudaSuccess) cudaHostUnregister(h_b);
    if (regC == cudaSuccess) cudaHostUnregister(h_c);

    return c;
}