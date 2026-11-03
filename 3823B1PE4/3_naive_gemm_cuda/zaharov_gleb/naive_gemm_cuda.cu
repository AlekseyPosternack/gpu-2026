#include "naive_gemm_cuda.h"

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

struct DeviceDeleter {
    void operator()(float* pointer) const noexcept {
        cudaFree(pointer);
    }
};

using DeviceBuffer = std::unique_ptr<float, DeviceDeleter>;

DeviceBuffer AllocateDevice(std::size_t bytes) {
    float* pointer = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&pointer), bytes), "cudaMalloc");
    return DeviceBuffer(pointer);
}

__global__ void NaiveGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                int n) {
    const std::size_t column = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t row = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (row >= static_cast<std::size_t>(n) || column >= static_cast<std::size_t>(n)) {
        return;
    }

    const std::size_t row_offset = row * n;
    float sum = 0.0f;
    // A warp computes consecutive columns: B loads and C stores are coalesced.
    // Each thread computes one element directly from global memory.
    #pragma unroll 4
    for (int k = 0; k < n; ++k) {
        sum = fmaf(a[row_offset + k], b[static_cast<std::size_t>(k) * n + column], sum);
    }
    c[row_offset + column] = sum;
}

} // namespace

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
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
    const auto device_a = AllocateDevice(bytes);
    const auto device_b = AllocateDevice(bytes);
    const auto device_c = AllocateDevice(bytes);
    CheckCuda(cudaMemcpy(device_a.get(), a.data(), bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy A host to device");
    CheckCuda(cudaMemcpy(device_b.get(), b.data(), bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy B host to device");

    const dim3 threads(32, 8);
    const dim3 blocks(static_cast<unsigned int>((dimension + threads.x - 1) / threads.x),
                      static_cast<unsigned int>((dimension + threads.y - 1) / threads.y));
    NaiveGemmKernel<<<blocks, threads>>>(device_a.get(), device_b.get(), device_c.get(), n);
    CheckCuda(cudaGetLastError(), "NaiveGemmKernel launch");

    std::vector<float> result(size);
    // The blocking copy waits for the kernel and reports execution errors.
    CheckCuda(cudaMemcpy(result.data(), device_c.get(), bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy C device to host");
    return result;
}
