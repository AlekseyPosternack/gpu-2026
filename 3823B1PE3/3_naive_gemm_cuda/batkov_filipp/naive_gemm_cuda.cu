#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

constexpr int ThreadsX = 32;
constexpr int ThreadsY = 4;
constexpr int RowsPerThread = 4;
constexpr int ColumnsPerThread = 4;

void CheckCudaError(const cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(error)
        );
    }
}

struct DeviceMemoryDeleter {
    void operator()(float* pointer) const noexcept {
        cudaFree(pointer);
    }
};

__global__ void NaiveGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                const int n) {
    const std::size_t row =
        static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const std::size_t column =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (row >= static_cast<std::size_t>(n) ||
        column >= static_cast<std::size_t>(n)) {
        return;
    }

    float sum = 0.0f;

    #pragma unroll 4
    for (int k = 0; k < n; ++k) {
        sum = fmaf(a[row * n + k],
                   b[static_cast<std::size_t>(k) * n + column], sum);
    }

    c[row * n + column] = sum;
}

__global__ void NaiveGemmVectorizedKernel(const float* __restrict__ a,
                                          const float4* __restrict__ b,
                                          float4* __restrict__ c,
                                          const int n) {
    const std::size_t row =
        (static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y) *
        RowsPerThread;
    const std::size_t vectorColumn =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t vectorStride = n / ColumnsPerThread;

    if (row >= static_cast<std::size_t>(n) || vectorColumn >= vectorStride) {
        return;
    }

    float4 sums[RowsPerThread] = {};

    #pragma unroll 4
    for (int k = 0; k < n; ++k) {
        const float4 valueB = b[static_cast<std::size_t>(k) * vectorStride +
                               vectorColumn];

        #pragma unroll
        for (int r = 0; r < RowsPerThread; ++r) {
            const float valueA = a[(row + r) * n + k];
            sums[r].x = fmaf(valueA, valueB.x, sums[r].x);
            sums[r].y = fmaf(valueA, valueB.y, sums[r].y);
            sums[r].z = fmaf(valueA, valueB.z, sums[r].z);
            sums[r].w = fmaf(valueA, valueB.w, sums[r].w);
        }
    }

    #pragma unroll
    for (int r = 0; r < RowsPerThread; ++r) {
        c[(row + r) * vectorStride + vectorColumn] = sums[r];
    }
}

} // namespace

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
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

    const std::size_t bytes = elementCount * sizeof(float);
    void* allocation = nullptr;

    CheckCudaError(cudaMalloc(&allocation, 3 * bytes),
                   "Failed to allocate device matrices");

    const std::unique_ptr<float, DeviceMemoryDeleter> deviceMemory(
        static_cast<float*>(allocation)
    );

    float* const deviceA = deviceMemory.get();
    float* const deviceB = deviceA + elementCount;
    float* const deviceC = deviceB + elementCount;

    CheckCudaError(cudaMemcpy(deviceA, a.data(), bytes, cudaMemcpyHostToDevice),
                   "Failed to copy A to GPU");
    CheckCudaError(cudaMemcpy(deviceB, b.data(), bytes, cudaMemcpyHostToDevice),
                   "Failed to copy B to GPU");

    const dim3 block(ThreadsX, ThreadsY);
    if (n >= 128 && n % ColumnsPerThread == 0) {

        const dim3 grid(
            static_cast<unsigned int>((size + ThreadsX * ColumnsPerThread - 1) /
                                      (ThreadsX * ColumnsPerThread)),
            static_cast<unsigned int>((size + ThreadsY * RowsPerThread - 1) /
                                      (ThreadsY * RowsPerThread))
        );

        NaiveGemmVectorizedKernel<<<grid, block>>>(
            deviceA,
            reinterpret_cast<const float4*>(deviceB),
            reinterpret_cast<float4*>(deviceC),
            n
        );
    } else {
        const dim3 grid(
            static_cast<unsigned int>((size + ThreadsX - 1) / ThreadsX),
            static_cast<unsigned int>((size + ThreadsY - 1) / ThreadsY)
        );
        NaiveGemmKernel<<<grid, block>>>(deviceA, deviceB, deviceC, n);
    }

    CheckCudaError(cudaGetLastError(), "Failed to launch GEMM kernel");

    std::vector<float> result(elementCount);
    CheckCudaError(cudaMemcpy(result.data(), deviceC, bytes, cudaMemcpyDeviceToHost),
                   "Failed to copy C from GPU");

    return result;
}
