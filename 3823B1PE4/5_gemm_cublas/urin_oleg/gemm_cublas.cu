#include "gemm_cublas.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void CheckCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void CheckCublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": cuBLAS status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

class DeviceStorage {
public:
    explicit DeviceStorage(std::size_t bytes) {
        void* memory = nullptr;
        CheckCuda(cudaMalloc(&memory, bytes), "cudaMalloc");
        data_ = static_cast<float*>(memory);
    }

    ~DeviceStorage() {
        cudaFree(data_);
    }

    DeviceStorage(const DeviceStorage&) = delete;
    DeviceStorage& operator=(const DeviceStorage&) = delete;

    float* data() const { return data_; }

private:
    float* data_ = nullptr;
};

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

    cublasHandle_t get() const { return handle_; }

private:
    cublasHandle_t handle_ = nullptr;
};

} // namespace

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (n < 0) {
        throw std::invalid_argument("n must be nonnegative");
    }

    const std::size_t count = static_cast<std::size_t>(n) * n;
    if (a.size() != count || b.size() != count) {
        throw std::invalid_argument("both matrices must contain n * n elements");
    }
    if (count == 0) {
        return {};
    }
    if (count > std::numeric_limits<std::size_t>::max() / (3 * sizeof(float))) {
        throw std::length_error("matrix size exceeds addressable device memory");
    }

    const std::size_t bytes = count * sizeof(float);
    DeviceStorage storage(3 * bytes);
    float* const device_a = storage.data();
    float* const device_b = device_a + count;
    float* const device_c = device_b + count;

    CheckCuda(cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy A to device");
    CheckCuda(cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy B to device");

    thread_local CublasHandle handle;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CheckCublas(cublasSgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N,
                            n, n, n, &alpha, device_b, n, device_a, n,
                            &beta, device_c, n),
                "cublasSgemm");

    std::vector<float> result(count);
    CheckCuda(cudaMemcpy(result.data(), device_c, bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy C to host");
    return result;
}
