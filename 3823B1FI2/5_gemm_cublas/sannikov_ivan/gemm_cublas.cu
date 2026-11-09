#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace {

struct GemmResources {
    cublasHandle_t blas = nullptr;
    cudaStream_t queue = nullptr;
    float* storage = nullptr;
    size_t capacity = 0;

    GemmResources() {
        cudaStreamCreateWithFlags(&queue, cudaStreamNonBlocking);
        cublasCreate(&blas);
        cublasSetStream(blas, queue);

#if defined(GEMM_ENABLE_TF32) && CUDART_VERSION >= 11000
        cublasSetMathMode(blas, CUBLAS_TF32_TENSOR_OP_MATH);
#endif
    }
    ~GemmResources() {
        cudaStreamSynchronize(queue);
        cudaFree(storage);
        cublasDestroy(blas);
        cudaStreamDestroy(queue);
    }
    void Reserve(size_t elements) {
        if (elements > capacity) {
            cudaFree(storage);
            cudaMalloc(reinterpret_cast<void**>(&storage), 3 * elements * sizeof(float));
            capacity = elements;
        }
    }
    GemmResources(const GemmResources&) = delete;
    GemmResources& operator=(const GemmResources&) = delete;
};
struct MatrixBuffers {
    float* left;
    float* right;
    float* result;
};

} // namespace

std::vector<float> GemmCUBLAS(const std::vector<float>& a, const std::vector<float>& b, int n) {
    if (n <= 0) {return {};}

    static thread_local GemmResources resources;
    const size_t elements = static_cast<size_t>(n) * n;
    const size_t bytes = elements * sizeof(float);
    resources.Reserve(elements);
    MatrixBuffers buffers{resources.storage, resources.storage + elements, resources.storage + 2 * elements};

    cudaMemcpyAsync(buffers.left, a.data(), bytes, cudaMemcpyHostToDevice, resources.queue);
    cudaMemcpyAsync(buffers.right, b.data(), bytes, cudaMemcpyHostToDevice, resources.queue);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(resources.blas, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, buffers.right, n, buffers.left, n, &beta, buffers.result, n);
    std::vector<float> result(elements);
    cudaMemcpyAsync(result.data(), buffers.result, bytes, cudaMemcpyDeviceToHost, resources.queue);
    cudaStreamSynchronize(resources.queue);
    return result;
}