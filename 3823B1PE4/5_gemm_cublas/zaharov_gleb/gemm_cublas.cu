#include "gemm_cublas.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <memory>
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
        throw std::runtime_error(std::string(operation) + ": cuBLAS status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

struct DeviceDeleter {
    void operator()(float* pointer) const noexcept {
        cudaFree(pointer);
    }
};

using DeviceBuffer = std::unique_ptr<float, DeviceDeleter>;

class CublasHandle {
public:
    CublasHandle() {
        CheckCublas(cublasCreate(&handle_), "cublasCreate");
    }

    ~CublasHandle() {
        cublasDestroy(handle_);
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    cublasHandle_t get() const noexcept { return handle_; }

private:
    cublasHandle_t handle_ = nullptr;
};

} // namespace

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (n < 0) {
        throw std::invalid_argument("Matrix dimension must be non-negative");
    }
    const std::size_t dimension = static_cast<std::size_t>(n);
    if (dimension != 0 &&
        dimension > std::numeric_limits<std::size_t>::max() / sizeof(float) / dimension) {
        throw std::length_error("Matrix size is too large");
    }
    const std::size_t size = dimension * dimension;
    if (a.size() != size || b.size() != size) {
        throw std::invalid_argument("Input matrices must contain n * n elements");
    }
    if (size == 0) {
        return {};
    }

    const std::size_t bytes = size * sizeof(float);
    if (bytes > std::numeric_limits<std::size_t>::max() / 3) {
        throw std::length_error("Matrix buffers are too large");
    }
    float* storage = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&storage), 3 * bytes), "cudaMalloc");
    const DeviceBuffer device_storage(storage);
    float* const device_a = storage;
    float* const device_b = storage + size;
    float* const device_c = storage + 2 * size;

    CheckCuda(cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy A host to device");
    CheckCuda(cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy B host to device");

    const CublasHandle handle;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // Row-major C = A * B has the same bytes as column-major C^T = B^T * A^T.
    CheckCublas(cublasSgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N,
                            n, n, n, &alpha, device_b, n, device_a, n,
                            &beta, device_c, n), "cublasSgemm");

    std::vector<float> result(size);
    // The blocking copy waits for cuBLAS execution and reports device errors.
    CheckCuda(cudaMemcpy(result.data(), device_c, bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy C device to host");
    return result;
}
