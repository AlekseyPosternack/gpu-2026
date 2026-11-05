#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

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

struct DeviceBuffers {
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;

    ~DeviceBuffers() {
        if (a != nullptr) {
            cudaFree(a);
        }
        if (b != nullptr) {
            cudaFree(b);
        }
        if (c != nullptr) {
            cudaFree(c);
        }
    }
};

__global__ void NaiveGemmKernel(
    const float* a,
    const float* b,
    float* c,
    int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        float sum = 0.0f;

        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }

        c[row * n + col] = sum;
    }
}

}  // namespace

std::vector<float> NaiveGemmCUDA(
    const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    if (n < 0) {
        throw std::invalid_argument("Matrix size cannot be negative");
    }

    const std::size_t elementCount =
        static_cast<std::size_t>(n) * static_cast<std::size_t>(n);

    if (a.size() != elementCount || b.size() != elementCount) {
        throw std::invalid_argument(
            "Input matrix sizes must be equal to n * n");
    }

    std::vector<float> c(elementCount);

    if (n == 0) {
        return c;
    }

    DeviceBuffers device;
    const std::size_t bytes = elementCount * sizeof(float);

    CheckCuda(cudaMalloc(&device.a, bytes), "cudaMalloc matrix A");
    CheckCuda(cudaMalloc(&device.b, bytes), "cudaMalloc matrix B");
    CheckCuda(cudaMalloc(&device.c, bytes), "cudaMalloc matrix C");

    CheckCuda(
        cudaMemcpy(device.a, a.data(), bytes, cudaMemcpyHostToDevice),
        "cudaMemcpy matrix A to device");

    CheckCuda(
        cudaMemcpy(device.b, b.data(), bytes, cudaMemcpyHostToDevice),
        "cudaMemcpy matrix B to device");

    constexpr int threadsPerBlock = 16;
    const dim3 threads(threadsPerBlock, threadsPerBlock);
    const dim3 blocks(
        (n + threadsPerBlock - 1) / threadsPerBlock,
        (n + threadsPerBlock - 1) / threadsPerBlock);

    NaiveGemmKernel<<<blocks, threads>>>(device.a, device.b, device.c, n);

    CheckCuda(cudaGetLastError(), "NaiveGemmKernel launch");

    CheckCuda(
        cudaMemcpy(c.data(), device.c, bytes, cudaMemcpyDeviceToHost),
        "cudaMemcpy result to host");

    return c;
}
