#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstddef>
#include <future>
#include <stdexcept>
#include <vector>

std::vector<float> GemmCUBLAS(
    const std::vector<float>& a,
    const std::vector<float>& b,
    int n
) {
    if (n <= 0)
        throw std::invalid_argument("Invalid n");

    size_t count = (size_t)n * n;

    if (a.size() != count || b.size() != count)
        throw std::invalid_argument("Invalid matrix size");

    size_t bytes = count * sizeof(float);

    float* buffer = nullptr;
    cublasHandle_t handle = nullptr;
    std::future<void> copy_future;

    auto checkCuda = [](cudaError_t error) {
        if (error != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(error));
        };

    auto checkCublas = [](cublasStatus_t status) {
        if (status != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cuBLAS error");
        };

    auto cleanup = [&]() {
        if (handle) cublasDestroy(handle);
        if (buffer) cudaFree(buffer);
        };

    try {
        checkCuda(cudaMalloc((void**)&buffer, bytes * 3));

        float* d_a = buffer;
        float* d_b = buffer + count;
        float* d_c = buffer + count * 2;

        checkCublas(cublasCreate(&handle));

        copy_future = std::async(std::launch::async, [&]() {
            checkCuda(cudaMemcpy(
                d_a, a.data(), bytes, cudaMemcpyHostToDevice
            ));

            checkCuda(cudaMemcpy(
                d_b, b.data(), bytes, cudaMemcpyHostToDevice
            ));

            checkCuda(cudaStreamSynchronize(0));
            });

        std::vector<float> result(count);

        copy_future.get();

        const float alpha = 1.0f;
        const float beta = 0.0f;

        checkCublas(cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &alpha,
            d_b, n,
            d_a, n,
            &beta,
            d_c, n
        ));

        checkCuda(cudaMemcpy(
            result.data(), d_c, bytes, cudaMemcpyDeviceToHost
        ));

        cleanup();
        return result;
    }
    catch (...) {
        if (copy_future.valid())
            copy_future.wait();

        cleanup();
        throw;
    }
}
