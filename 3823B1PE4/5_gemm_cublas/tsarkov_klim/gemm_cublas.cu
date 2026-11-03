#include "gemm_cublas.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void CheckCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void CheckCublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + ": cuBLAS error " +
            std::to_string(static_cast<int>(status)));
    }
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) {
        CheckCuda(cudaMalloc(&ptr_, bytes), "cudaMalloc");
    }

    ~DeviceBuffer() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    float* get() const {
        return ptr_;
    }

private:
    float* ptr_ = nullptr;
};

class CublasHandle {
public:
    CublasHandle() {
        CheckCublas(cublasCreate(&handle_), "cublasCreate");
    }

    ~CublasHandle() {
        if (handle_ != nullptr) {
            cublasDestroy(handle_);
        }
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    cublasHandle_t get() const {
        return handle_;
    }

private:
    cublasHandle_t handle_ = nullptr;
};

} // namespace

std::vector<float> GemmCUBLAS(
    const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    if (n <= 0) {
        throw std::invalid_argument("Matrix size must be positive");
    }

    const std::size_t matrixSize =
        static_cast<std::size_t>(n) * static_cast<std::size_t>(n);

    if (a.size() != matrixSize || b.size() != matrixSize) {
        throw std::invalid_argument(
            "Input vectors must each contain n * n elements");
    }

    std::vector<float> c(matrixSize);
    const std::size_t bytes = matrixSize * sizeof(float);

    DeviceBuffer deviceA(bytes);
    DeviceBuffer deviceB(bytes);
    DeviceBuffer deviceC(bytes);

    CheckCuda(
        cudaMemcpy(deviceA.get(), a.data(), bytes, cudaMemcpyHostToDevice),
        "Copy matrix A to device");
    CheckCuda(
        cudaMemcpy(deviceB.get(), b.data(), bytes, cudaMemcpyHostToDevice),
        "Copy matrix B to device");

    CublasHandle handle;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS uses column-major storage; reverse A and B to get
    // the expected row-major result.
    CheckCublas(
        cublasSgemm(
            handle.get(),
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            n,
            n,
            n,
            &alpha,
            deviceB.get(),
            n,
            deviceA.get(),
            n,
            &beta,
            deviceC.get(),
            n),
        "cublasSgemm");

    CheckCuda(cudaDeviceSynchronize(), "Run cublasSgemm");
    CheckCuda(
        cudaMemcpy(c.data(), deviceC.get(), bytes, cudaMemcpyDeviceToHost),
        "Copy result to host");

    return c;
}
