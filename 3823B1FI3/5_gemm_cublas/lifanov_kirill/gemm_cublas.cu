#include "gemm_cublas.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

struct CudaContextManager {
    cublasHandle_t blas_handle;
    cudaStream_t cuda_stream;
    float* dev_a;
    float* dev_b;
    float* dev_c;
    size_t capacity;

    CudaContextManager()
        : blas_handle(nullptr), cuda_stream(nullptr),
        dev_a(nullptr), dev_b(nullptr), dev_c(nullptr), capacity(0) {

        cublasCreate(&blas_handle);
        cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking);
        cublasSetStream(blas_handle, cuda_stream);
    }

    ~CudaContextManager() {
        if (dev_a) cudaFree(dev_a);
        if (dev_b) cudaFree(dev_b);
        if (dev_c) cudaFree(dev_c);
        if (cuda_stream) cudaStreamDestroy(cuda_stream);
        if (blas_handle) cublasDestroy(blas_handle);
    }

    bool AllocateIfNeeded(size_t elements_count) {
        if (capacity >= elements_count) return true;

        if (dev_a) cudaFree(dev_a);
        if (dev_b) cudaFree(dev_b);
        if (dev_c) cudaFree(dev_c);

        size_t bytes = elements_count * sizeof(float);
        if (cudaMalloc(&dev_a, bytes) != cudaSuccess) return false;
        if (cudaMalloc(&dev_b, bytes) != cudaSuccess) return false;
        if (cudaMalloc(&dev_c, bytes) != cudaSuccess) return false;

        capacity = elements_count;
        return true;
    }
};

static CudaContextManager g_cuda_manager;

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    if (n <= 0) return {};

    size_t num_elements = static_cast<size_t>(n) * n;
    if (!g_cuda_manager.AllocateIfNeeded(num_elements)) return {};

    size_t data_size = num_elements * sizeof(float);
    std::vector<float> result(num_elements);

    cudaMemcpyAsync(g_cuda_manager.dev_a, a.data(), data_size, cudaMemcpyHostToDevice, g_cuda_manager.cuda_stream);
    cudaMemcpyAsync(g_cuda_manager.dev_b, b.data(), data_size, cudaMemcpyHostToDevice, g_cuda_manager.cuda_stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(g_cuda_manager.blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n,
        &alpha,
        g_cuda_manager.dev_b, n,
        g_cuda_manager.dev_a, n,
        &beta,
        g_cuda_manager.dev_c, n);

    cudaMemcpyAsync(result.data(), g_cuda_manager.dev_c, data_size, cudaMemcpyDeviceToHost, g_cuda_manager.cuda_stream);

    cudaStreamSynchronize(g_cuda_manager.cuda_stream);

    return result;
}