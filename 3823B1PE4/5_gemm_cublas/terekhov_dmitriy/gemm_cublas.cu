#include "gemm_cublas.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace {

void checkCuda(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

void checkCublas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS error: " + std::to_string(static_cast<int>(status)));
    }
}

}

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                            const std::vector<float>& b,
                            int n) {
    if (n < 0) {
        throw std::invalid_argument("Matrix size must be nonnegative");
    }
    const std::size_t size = static_cast<std::size_t>(n) * n;
    if (a.size() != size || b.size() != size) {
        throw std::invalid_argument("Matrix dimensions do not match the data");
    }
    if (n == 0) {
        return {};
    }

    const std::size_t bytes = size * sizeof(float);
    float* device_data = nullptr;
    checkCuda(cudaMalloc(&device_data, 3 * bytes));
    float* device_a = device_data;
    float* device_b = device_data + size;
    float* device_c = device_data + 2 * size;

    cublasHandle_t handle = nullptr;
    std::vector<float> result;
    try {
        checkCublas(cublasCreate(&handle));
        checkCuda(cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice));

        const float alpha = 1.0f;
        const float beta = 0.0f;
        checkCublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                               n, n, n, &alpha, device_b, n, device_a, n,
                               &beta, device_c, n));

        result.resize(size);
        checkCuda(cudaMemcpy(result.data(), device_c, bytes, cudaMemcpyDeviceToHost));
    } catch (...) {
        if (handle != nullptr) {
            cublasDestroy(handle);
        }
        cudaFree(device_data);
        throw;
    }

    const cublasStatus_t status = cublasDestroy(handle);
    const cudaError_t error = cudaFree(device_data);
    checkCublas(status);
    checkCuda(error);
    return result;
}
