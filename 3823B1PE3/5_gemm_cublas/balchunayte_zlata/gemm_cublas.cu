#include "gemm_cublas.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <cstddef>
#include <limits>

#define CUDA_CHECK(call) do {                              \
    cudaError_t err = (call);                              \
    if (err != cudaSuccess)                                \
        throw std::runtime_error(cudaGetErrorString(err)); \
} while (0)

#define CUBLAS_CHECK(call) do {                             \
    cublasStatus_t status = (call);                          \
    if (status != CUBLAS_STATUS_SUCCESS)                     \
        throw std::runtime_error("cuBLAS operation failed"); \
} while (0)

// Память и cuBLAS-контекст для повторного использования
struct Workspace {
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    size_t capacity = 0;
    cublasHandle_t handle = nullptr;

    Workspace() {
        CUBLAS_CHECK(cublasCreate(&handle));
        CUBLAS_CHECK(cublasSetMathMode(
            handle, CUBLAS_DEFAULT_MATH
        ));
    }

    void reserve(size_t bytes) {
        if (bytes <= capacity) return;

        // Сначала освобождаем старую память
        if (d_a) CUDA_CHECK(cudaFree(d_a));
        d_a = nullptr;
        if (d_b) CUDA_CHECK(cudaFree(d_b));
        d_b = nullptr;
        if (d_c) CUDA_CHECK(cudaFree(d_c));
        d_c = nullptr;
        capacity = 0;

        CUDA_CHECK(cudaMalloc(&d_a, bytes));
        CUDA_CHECK(cudaMalloc(&d_b, bytes));
        CUDA_CHECK(cudaMalloc(&d_c, bytes));

        capacity = bytes;
    }

    ~Workspace() {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
        if (handle) cublasDestroy(handle);
    }
};

std::vector<float> GemmCUBLAS(
    const std::vector<float>& a,
    const std::vector<float>& b,
    int n
) {
    if (n <= 0 ||
        static_cast<size_t>(n) * n != a.size() ||
        a.size() != b.size()) {
        throw std::invalid_argument("Invalid matrix size");
    }

    size_t bytes = a.size() * sizeof(float);
    std::vector<float> result(a.size());

    // Для каждого CPU-потока свой кеш памяти GPU
    static thread_local Workspace ws;

    ws.reserve(bytes);

    CUDA_CHECK(cudaMemcpy(
        ws.d_a, a.data(), bytes, cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        ws.d_b, b.data(), bytes, cudaMemcpyHostToDevice
    ));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // B * A для корректной row-major матрицы C
    CUBLAS_CHECK(cublasSgemm(
        ws.handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n, n, n,
        &alpha,
        ws.d_b, n,
        ws.d_a, n,
        &beta,
        ws.d_c, n
    ));

    CUDA_CHECK(cudaMemcpy(
        result.data(), ws.d_c, bytes,
        cudaMemcpyDeviceToHost
    ));

    return result;
}
