#include "gemm_cublas.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

void CheckCudaError(const cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(error)
        );
    }
}

void CheckCublasError(const cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + ": cuBLAS status " +
            std::to_string(static_cast<int>(status))
        );
    }
}

struct DeviceMemoryDeleter {
    void operator()(float* pointer) const noexcept {
        cudaFree(pointer);
    }
};

class GemmWorkspace {
public:
    explicit GemmWorkspace(const int device) : device(device) {
        CheckCublasError(cublasCreate(&handle),
                         "Failed to create cuBLAS handle");
    }

    ~GemmWorkspace() noexcept {
        int previousDevice = -1;
        if (cudaGetDevice(&previousDevice) != cudaSuccess) {
            memory.release();
            return;
        }

        const bool switchDevice = previousDevice != device;
        if (switchDevice && cudaSetDevice(device) != cudaSuccess) {
            memory.release();
            return;
        }

        cublasDestroy(handle);
        memory.reset();
        if (switchDevice) {
            cudaSetDevice(previousDevice);
        }
    }

    GemmWorkspace(const GemmWorkspace&) = delete;
    GemmWorkspace& operator=(const GemmWorkspace&) = delete;

    void Reserve(const std::size_t elementCount) {
        if (elementCount <= capacity) {
            return;
        }

        void* allocation = nullptr;
        CheckCudaError(cudaMalloc(&allocation, 3 * elementCount * sizeof(float)),
                       "Failed to allocate device matrices");
        std::unique_ptr<float, DeviceMemoryDeleter> nextMemory(
            static_cast<float*>(allocation)
        );

        memory.swap(nextMemory);
        capacity = elementCount;
    }

    const int device;
    cublasHandle_t handle = nullptr;
    std::size_t capacity = 0;
    std::unique_ptr<float, DeviceMemoryDeleter> memory;
};

GemmWorkspace& GetWorkspace() {
    int device = -1;
    CheckCudaError(cudaGetDevice(&device), "Failed to get current CUDA device");

    thread_local std::unique_ptr<GemmWorkspace> workspace;
    if (!workspace || workspace->device != device) {
        auto nextWorkspace = std::make_unique<GemmWorkspace>(device);
        workspace.swap(nextWorkspace);
    }

    return *workspace;
}

} // namespace

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                            const std::vector<float>& b,
                            const int n) {
    if (n < 0) {
        throw std::invalid_argument("Matrix size must be non-negative");
    }

    const std::size_t size = static_cast<std::size_t>(n);
    const std::size_t maxSize = std::numeric_limits<std::size_t>::max();
    if (size != 0 && size > maxSize / size) {
        throw std::length_error("Matrix size is too large");
    }

    const std::size_t elementCount = size * size;
    if (a.size() != elementCount || b.size() != elementCount) {
        throw std::invalid_argument("Both matrices must contain n*n elements");
    }
    if (elementCount == 0) {
        return {};
    }
    if (elementCount > maxSize / sizeof(float) / 3) {
        throw std::length_error("Device allocation size is too large");
    }

    GemmWorkspace& workspace = GetWorkspace();
    workspace.Reserve(elementCount);

    float* const deviceA = workspace.memory.get();
    float* const deviceB = deviceA + workspace.capacity;
    float* const deviceC = deviceB + workspace.capacity;
    const std::size_t bytes = elementCount * sizeof(float);

    CheckCudaError(cudaMemcpy(deviceA, a.data(), bytes, cudaMemcpyHostToDevice),
                   "Failed to copy A to GPU");
    CheckCudaError(cudaMemcpy(deviceB, b.data(), bytes, cudaMemcpyHostToDevice),
                   "Failed to copy B to GPU");

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CheckCublasError(
        cublasSgemm(workspace.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    n, n, n, &alpha, deviceB, n, deviceA, n,
                    &beta, deviceC, n),
        "Failed to perform cuBLAS SGEMM"
    );

    std::vector<float> result(elementCount);

    CheckCudaError(cudaMemcpy(result.data(), deviceC, bytes, cudaMemcpyDeviceToHost),
                   "Failed to copy C from GPU");

    return result;
}
