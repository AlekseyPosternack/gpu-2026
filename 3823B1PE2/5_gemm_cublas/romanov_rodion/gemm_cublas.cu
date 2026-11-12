#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>

class CublasContext {
public:
    cublasHandle_t handle;
    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;
    size_t current_capacity = 0;

    CublasContext() {
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("CUBLAS initialization failed");
        }
    }

    ~CublasContext() {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
        cublasDestroy(handle);
    }

    void ensure_capacity(size_t num_elements) {
        size_t bytes = num_elements * sizeof(float);
        if (bytes > current_capacity) {
            if (d_a) cudaFree(d_a);
            if (d_b) cudaFree(d_b);
            if (d_c) cudaFree(d_c);

            cudaMalloc(&d_a, bytes);
            cudaMalloc(&d_b, bytes);
            cudaMalloc(&d_c, bytes);
            current_capacity = bytes;
        }
    }

    static CublasContext& getInstance() {
        static CublasContext instance;
        return instance;
    }
};

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    size_t total_elements = n * n;
    size_t size_bytes = total_elements * sizeof(float);
    std::vector<float> c(total_elements);

    auto& ctx = CublasContext::getInstance();
    ctx.ensure_capacity(total_elements);

    cudaMemcpy(ctx.d_a, a.data(), size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx.d_b, b.data(), size_bytes, cudaMemcpyHostToDevice);

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasStatus_t status = cublasSgemm(
        ctx.handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n, n, n,
        &alpha,
        ctx.d_b, n,
        ctx.d_a, n,
        &beta,
        ctx.d_c, n
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("CUBLAS GEMM execution failed");
    }

    cudaMemcpy(c.data(), ctx.d_c, size_bytes, cudaMemcpyDeviceToHost);

    return c;
}