#include "gemm_cublas.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace {

void CheckCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

void CheckCublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) +
                                 " failed with cuBLAS status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

} // namespace

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (n < 0) {
        throw std::invalid_argument("Matrix size n must be non-negative");
    }

    const std::size_t dimension = static_cast<std::size_t>(n);
    if (dimension != 0 &&
        dimension > std::numeric_limits<std::size_t>::max() / dimension) {
        throw std::invalid_argument("Matrix size is too large");
    }
    const std::size_t elements = dimension * dimension;
    if (a.size() != elements || b.size() != elements) {
        throw std::invalid_argument("Input matrices must have size n*n");
    }

    std::vector<float> result(elements);
    if (elements == 0) {
        return result;
    }

    float* deviceA = nullptr;
    float* deviceB = nullptr;
    float* deviceC = nullptr;
    cublasHandle_t handle = nullptr;
    const std::size_t bytes = elements * sizeof(float);

    try {
        // The device buffers are allocated once and reused for the complete
        // operation. No explicit transpose is needed: row-major A and B are
        // column-major A^T and B^T when viewed by cuBLAS, so
        // B^T * A^T = (A * B)^T produces the desired row-major result.
        CheckCuda(cudaMalloc(&deviceA, bytes), "cudaMalloc(A)");
        CheckCuda(cudaMalloc(&deviceB, bytes), "cudaMalloc(B)");
        CheckCuda(cudaMalloc(&deviceC, bytes), "cudaMalloc(C)");

        CheckCuda(cudaMemcpy(deviceA, a.data(), bytes, cudaMemcpyHostToDevice),
                  "cudaMemcpy(A)");
        CheckCuda(cudaMemcpy(deviceB, b.data(), bytes, cudaMemcpyHostToDevice),
                  "cudaMemcpy(B)");

        CheckCublas(cublasCreate(&handle), "cublasCreate");

        constexpr float alpha = 1.0f;
        constexpr float beta = 0.0f;
        CheckCublas(cublasSgemm(handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                n, n, n,
                                &alpha,
                                deviceB, n,
                                deviceA, n,
                                &beta,
                                deviceC, n),
                    "cublasSgemm");

        CheckCuda(cudaMemcpy(result.data(), deviceC, bytes,
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(C)");

        CheckCublas(cublasDestroy(handle), "cublasDestroy");
        handle = nullptr;
        CheckCuda(cudaFree(deviceC), "cudaFree(C)");
        deviceC = nullptr;
        CheckCuda(cudaFree(deviceB), "cudaFree(B)");
        deviceB = nullptr;
        CheckCuda(cudaFree(deviceA), "cudaFree(A)");
        deviceA = nullptr;
    } catch (...) {
        if (handle != nullptr) {
            cublasDestroy(handle);
        }
        if (deviceC != nullptr) {
            cudaFree(deviceC);
        }
        if (deviceB != nullptr) {
            cudaFree(deviceB);
        }
        if (deviceA != nullptr) {
            cudaFree(deviceA);
        }
        throw;
    }

    return result;
}
