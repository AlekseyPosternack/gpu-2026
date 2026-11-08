#include "gemm_cublas.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace {

void CheckCUDA(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(error)
        );
    }
}

void CheckCUBLAS(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + ": cuBLAS error " +
            std::to_string(static_cast<int>(status))
        );
    }
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) {
        CheckCUDA(
            cudaMalloc(reinterpret_cast<void**>(&ptr), bytes),
            "cudaMalloc"
        );
    }

    ~DeviceBuffer() {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    float* get() const {
        return ptr;
    }

private:
    float* ptr = nullptr;
};

class CublasHandle {
public:
    CublasHandle() {
        CheckCUBLAS(cublasCreate(&handle), "cublasCreate");
    }

    ~CublasHandle() {
        if (handle != nullptr) {
            cublasDestroy(handle);
        }
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    cublasHandle_t get() const {
        return handle;
    }

private:
    cublasHandle_t handle = nullptr;
};

} // namespace

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (n < 0) {
        throw std::invalid_argument("Matrix size cannot be negative");
    }

    const std::size_t count =
        static_cast<std::size_t>(n) * static_cast<std::size_t>(n);

    if (a.size() != count || b.size() != count) {
        throw std::invalid_argument(
            "Matrices must have exactly n*n elements"
        );
    }

    std::vector<float> c(count, 0.0f);

    if (n == 0) {
        return c;
    }

    const std::size_t bytes = count * sizeof(float);

    DeviceBuffer deviceMemory(3 * bytes);

    float* d_a = deviceMemory.get();
    float* d_b = d_a + count;
    float* d_c = d_b + count;

    CheckCUDA(
        cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice),
        "cudaMemcpy A"
    );

    CheckCUDA(
        cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice),
        "cudaMemcpy B"
    );

    CublasHandle handle;

    const float alpha = 1.0f;
    const float beta = 0.0f;


    CheckCUBLAS(
        cublasSgemm(
            handle.get(),
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            n, n, n,
            &alpha,
            d_b, n,
            d_a, n,
            &beta,
            d_c, n
        ),
        "cublasSgemm"
    );

    CheckCUDA(
        cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost),
        "cudaMemcpy C"
    );

    return c;
}