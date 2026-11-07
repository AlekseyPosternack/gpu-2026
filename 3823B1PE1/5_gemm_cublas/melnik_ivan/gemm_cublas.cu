#include "gemm_cublas.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstddef>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (n <= 0) {
        return {};
    }

    const std::size_t total_elems = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    const std::size_t matrix_bytes = total_elems * sizeof(float);
    
    static thread_local cublasHandle_t blas_handle = nullptr;
    static thread_local cudaStream_t work_stream = nullptr;
    static thread_local float* dev_slab = nullptr;
    static thread_local std::size_t allocated_elems = 0;

    if (blas_handle == nullptr) {
        cudaStreamCreateWithFlags(&work_stream, cudaStreamNonBlocking);
        cublasCreate(&blas_handle);
        cublasSetStream(blas_handle, work_stream);
        cublasSetPointerMode(blas_handle, CUBLAS_POINTER_MODE_HOST);
    }

    if (total_elems > allocated_elems) {
        if (dev_slab != nullptr) {
            cudaFree(dev_slab);
        }
        cudaMalloc(&dev_slab, 3 * matrix_bytes);
        allocated_elems = total_elems;
    }

    float* const d_a = dev_slab;
    float* const d_b = dev_slab + total_elems;
    float* const d_c = dev_slab + 2 * total_elems;

    cudaMemcpyAsync(d_a, a.data(), matrix_bytes, cudaMemcpyHostToDevice, work_stream);
    cudaMemcpyAsync(d_b, b.data(), matrix_bytes, cudaMemcpyHostToDevice, work_stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(blas_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha,
                d_b, n,
                d_a, n,
                &beta,
                d_c, n);

    cudaStreamQuery(work_stream);

    std::vector<float> c(total_elems);

    cudaMemcpyAsync(c.data(), d_c, matrix_bytes, cudaMemcpyDeviceToHost, work_stream);
    cudaStreamSynchronize(work_stream);

    return c;
}