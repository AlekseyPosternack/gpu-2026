#include "gemm_cublas.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace {

void CheckCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

void CheckCublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with status " + std::to_string(static_cast<int>(status)));
    }
}

class CublasResources {
public:
    CublasResources() {
        CheckCublas(cublasCreate(&handle_), "cublasCreate");
    }

    ~CublasResources() {
        cudaFree(data_);
        cublasDestroy(handle_);
    }

    float* Reserve(std::size_t size) {
        if (size <= capacity_) {
            return data_;
        }
        if (data_ != nullptr) {
            CheckCuda(cudaFree(data_), "cudaFree");
            data_ = nullptr;
            capacity_ = 0;
        }
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&data_), size * sizeof(float)), "cudaMalloc");
        capacity_ = size;
        return data_;
    }

    cublasHandle_t Handle() const {
        return handle_;
    }

private:
    cublasHandle_t handle_ = nullptr;
    float* data_ = nullptr;
    std::size_t capacity_ = 0;
};

}

std::vector<float> GemmCUBLAS(const std::vector<float>& a, const std::vector<float>& b, int n) {
    if (n < 0) {
        throw std::invalid_argument("matrix size must not be negative");
    }
    const std::size_t count = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    if (a.size() != count || b.size() != count) {
        throw std::invalid_argument("matrix dimensions do not match n");
    }
    if (count == 0) {
        return {};
    }

    const std::size_t bytes = count * sizeof(float);
    static thread_local CublasResources resources;
    float* deviceA = resources.Reserve(count * 3);
    float* deviceB = deviceA + count;
    float* deviceC = deviceB + count;

    CheckCuda(cudaMemcpy(deviceA, a.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy a");
    CheckCuda(cudaMemcpy(deviceB, b.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy b");

    const float alpha = 1.0f;
    const float beta = 0.0f;
    CheckCublas(cublasSgemm(resources.Handle(), CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, deviceB, n, deviceA, n, &beta, deviceC, n), "cublasSgemm");

    std::vector<float> result(count);
    CheckCuda(cudaMemcpy(result.data(), deviceC, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy result");
    return result;
}
