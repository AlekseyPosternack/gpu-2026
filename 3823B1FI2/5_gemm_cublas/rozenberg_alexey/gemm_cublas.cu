#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace {
struct CublasContext {
    cublasHandle_t handle = nullptr;
    cudaStream_t stream = nullptr;
    float* d_mem = nullptr;
    size_t allocated_bytes = 0;

    CublasContext() {
        cublasCreate(&handle);
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        cublasSetStream(handle, stream);
        
        // Enable Tensor Core acceleration for FP32 SGEMM (Ampere+ GPUs)
        cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
    }

    ~CublasContext() {
        if (d_mem) cudaFree(d_mem);
        if (handle) cublasDestroy(handle);
        if (stream) cudaStreamDestroy(stream);
    }

    float* get_buffers(size_t bytes_per_matrix) {
        size_t total_needed = 3 * bytes_per_matrix;
        if (total_needed > allocated_bytes) {
            if (d_mem) cudaFree(d_mem);
            cudaMalloc(&d_mem, total_needed);
            allocated_bytes = total_needed;
        }
        return d_mem;
    }
};
} // namespace

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (n <= 0) return {};

    static CublasContext ctx;

    size_t num_elements = static_cast<size_t>(n) * n;
    size_t bytes = num_elements * sizeof(float);

    float* d_mem = ctx.get_buffers(bytes);
    float* d_a = d_mem;
    float* d_b = d_mem + num_elements;
    float* d_c = d_mem + 2 * num_elements;

    cudaMemcpyAsync(d_a, a.data(), bytes, cudaMemcpyHostToDevice, ctx.stream);
    cudaMemcpyAsync(d_b, b.data(), bytes, cudaMemcpyHostToDevice, ctx.stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(ctx.handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha,
                d_b, n,
                d_a, n,
                &beta,
                d_c, n);

    std::vector<float> c(num_elements);

    cudaMemcpyAsync(c.data(), d_c, bytes, cudaMemcpyDeviceToHost, ctx.stream);
    cudaStreamSynchronize(ctx.stream);

    return c;
}