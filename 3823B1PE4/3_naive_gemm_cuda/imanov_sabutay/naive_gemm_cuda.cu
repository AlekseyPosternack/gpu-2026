#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void CheckCUDA(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

struct DeviceMemory {
    float* data = nullptr;

    DeviceMemory() = default;
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;

    ~DeviceMemory() {
        if (data != nullptr) cudaFree(data);
    }
};

__global__ void NaiveGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c, int n) {
    const std::size_t row = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const std::size_t col = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= static_cast<std::size_t>(n) || col >= static_cast<std::size_t>(n)) return;

    const std::size_t row_offset = row * n;
    float sum = 0.0f;
    // Each warp computes adjacent columns: B loads and C stores are contiguous.
    // All lanes in a warp read the same element of A.
    #pragma unroll 4
    for (int k = 0; k < n; ++k) {
        sum = fmaf(a[row_offset + k], b[static_cast<std::size_t>(k) * n + col], sum);
    }
    c[row_offset + col] = sum;
}

} // namespace

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                               const std::vector<float>& b,
                               int n) {
    if (n < 0) throw std::invalid_argument("matrix size must be nonnegative");
    const std::size_t side = static_cast<std::size_t>(n);
    const std::size_t max_elements = std::numeric_limits<std::size_t>::max() / (3 * sizeof(float));
    if (side != 0 && side > max_elements / side) {
        throw std::length_error("matrix size is too large");
    }
    const std::size_t elements = side * side;
    if (a.size() != elements || b.size() != elements) {
        throw std::invalid_argument("input matrices must contain n*n elements");
    }
    if (elements == 0) return {};

    const std::size_t bytes = elements * sizeof(float);
    DeviceMemory memory;
    CheckCUDA(cudaMalloc(reinterpret_cast<void**>(&memory.data), 3 * bytes), "cudaMalloc");
    float* device_a = memory.data;
    float* device_b = device_a + elements;
    float* device_c = device_b + elements;
    CheckCUDA(cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice), "copy A to GPU");
    CheckCUDA(cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice), "copy B to GPU");

    const dim3 threads(32, 8);
    const dim3 blocks(static_cast<unsigned int>((side + threads.x - 1) / threads.x),
                      static_cast<unsigned int>((side + threads.y - 1) / threads.y));
    NaiveGemmKernel<<<blocks, threads>>>(device_a, device_b, device_c, n);
    CheckCUDA(cudaGetLastError(), "NaiveGemmKernel launch");

    // Allocate the output while the kernel may still be running.
    std::vector<float> output(elements);
    // Blocking device-to-host copy waits for the kernel to finish.
    CheckCUDA(cudaMemcpy(output.data(), device_c, bytes, cudaMemcpyDeviceToHost), "copy C from GPU");
    return output;
}
