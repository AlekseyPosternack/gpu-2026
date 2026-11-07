#include "block_gemm_cuda.h"

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

constexpr unsigned int kTile = 16;

__global__ void BlockGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c, int n) {
    __shared__ float tile_a[kTile][kTile];
    __shared__ float tile_b[kTile][kTile];
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const std::size_t side = static_cast<std::size_t>(n);
    const std::size_t row = static_cast<std::size_t>(blockIdx.y) * kTile + ty;
    const std::size_t col = static_cast<std::size_t>(blockIdx.x) * kTile + tx;
    // Keep each thread's partial result in a register.
    float sum = 0.0f;

    for (std::size_t base = 0; base < side; base += kTile) {
        const std::size_t a_col = base + tx;
        const std::size_t b_row = base + ty;
        tile_a[ty][tx] = (row < side && a_col < side) ? a[row * side + a_col] : 0.0f;
        tile_b[ty][tx] = (b_row < side && col < side) ? b[b_row * side + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (unsigned int k = 0; k < kTile; ++k) {
            sum = fmaf(tile_a[ty][k], tile_b[k][tx], sum);
        }
        // Every thread must finish reading before the next tile is loaded.
        __syncthreads();
    }

    if (row < side && col < side) c[row * side + col] = sum;
}
} // namespace

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
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

    const dim3 threads(kTile, kTile);
    const dim3 blocks(static_cast<unsigned int>((side + threads.x - 1) / threads.x),
                      static_cast<unsigned int>((side + threads.y - 1) / threads.y));
    BlockGemmKernel<<<blocks, threads>>>(device_a, device_b, device_c, n);
    CheckCUDA(cudaGetLastError(), "BlockGemmKernel launch");

    // Allocate the output while the kernel may still be running.
    std::vector<float> output(elements);
    // Blocking device-to-host copy waits for the kernel to finish.
    CheckCUDA(cudaMemcpy(output.data(), device_c, bytes, cudaMemcpyDeviceToHost), "copy C from GPU");
    return output;
}
