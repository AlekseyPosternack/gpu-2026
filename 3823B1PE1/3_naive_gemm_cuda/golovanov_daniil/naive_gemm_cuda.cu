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
        if (a != nullptr) cudaFree(a);
        if (b != nullptr) cudaFree(b);
        if (c != nullptr) cudaFree(c);
    }
};

__global__ void NaiveGemmKernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= n || col >= n) return;

    float sum = 0.0f;
    const float* a_row = a + static_cast<std::size_t>(row) * n;
    const float* b_col = b + col;

    #pragma unroll 8
    for (int k = 0; k < n; ++k) {
        sum += a_row[k] * b_col[static_cast<std::size_t>(k) * n];
    }

    c[static_cast<std::size_t>(row) * n + col] = sum;
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

    cudaStream_t stream = nullptr;
    CheckCuda(cudaStreamCreate(&stream), "cudaStreamCreate");

    CheckCuda(
        cudaMemcpyAsync(device.a, a.data(), bytes,
                        cudaMemcpyHostToDevice, stream),
        "cudaMemcpyAsync matrix A to device");

    CheckCuda(
        cudaMemcpyAsync(device.b, b.data(), bytes,
                        cudaMemcpyHostToDevice, stream),
        "cudaMemcpyAsync matrix B to device");

    constexpr int threadsPerBlock = 16;
    const dim3 threads(threadsPerBlock, threadsPerBlock);
    const dim3 blocks(
        (n + threadsPerBlock - 1) / threadsPerBlock,
        (n + threadsPerBlock - 1) / threadsPerBlock);

    NaiveGemmKernel<<<blocks, threads, 0, stream>>>(
        device.a, device.b, device.c, n);

    CheckCuda(cudaGetLastError(), "NaiveGemmKernel launch");

    CheckCuda(
        cudaMemcpyAsync(c.data(), device.c, bytes,
                        cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync result to host");

    CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    CheckCuda(cudaStreamDestroy(stream), "cudaStreamDestroy");

    return c;
}