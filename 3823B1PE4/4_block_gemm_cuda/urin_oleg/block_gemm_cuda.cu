#include "block_gemm_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace {

constexpr int kTileSize = 16;

void CheckCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) {
        void* memory = nullptr;
        CheckCuda(cudaMalloc(&memory, bytes), "cudaMalloc");
        data_ = static_cast<float*>(memory);
    }

    ~DeviceBuffer() {
        cudaFree(data_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    float* data() const { return data_; }

private:
    float* data_ = nullptr;
};

__global__ void BlockGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                int n) {
    __shared__ float tile_a[kTileSize][kTileSize];
    __shared__ float tile_b[kTileSize][kTileSize];
    __shared__ float tile_c[kTileSize][kTileSize];

    const int local_row = threadIdx.y;
    const int local_col = threadIdx.x;
    const int row = blockIdx.y * kTileSize + local_row;
    const int col = blockIdx.x * kTileSize + local_col;
    tile_c[local_row][local_col] = 0.0f;

    for (int tile_start = 0; tile_start < n; tile_start += kTileSize) {
        const int a_col = tile_start + local_col;
        const int b_row = tile_start + local_row;
        tile_a[local_row][local_col] =
            (row < n && a_col < n) ? a[static_cast<std::size_t>(row) * n + a_col] : 0.0f;
        tile_b[local_row][local_col] =
            (b_row < n && col < n) ? b[static_cast<std::size_t>(b_row) * n + col] : 0.0f;
        __syncthreads();

        float sum = tile_c[local_row][local_col];
#pragma unroll
        for (int k = 0; k < kTileSize; ++k) {
            sum = fmaf(tile_a[local_row][k], tile_b[k][local_col], sum);
        }
        tile_c[local_row][local_col] = sum;
        __syncthreads();
    }

    if (row < n && col < n) {
        c[static_cast<std::size_t>(row) * n + col] = tile_c[local_row][local_col];
    }
}

} // namespace

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
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

    const std::size_t bytes = count * sizeof(float);
    DeviceBuffer device_a(bytes);
    DeviceBuffer device_b(bytes);
    DeviceBuffer device_c(bytes);

    CheckCuda(cudaMemcpy(device_a.data(), a.data(), bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy A to device");
    CheckCuda(cudaMemcpy(device_b.data(), b.data(), bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy B to device");

    const dim3 threads(kTileSize, kTileSize);
    const unsigned int blocks_per_side =
        (static_cast<unsigned int>(n) + kTileSize - 1) / kTileSize;
    const dim3 blocks(blocks_per_side, blocks_per_side);
    BlockGemmKernel<<<blocks, threads>>>(device_a.data(), device_b.data(), device_c.data(), n);
    CheckCuda(cudaGetLastError(), "BlockGemmKernel launch");

    std::vector<float> result(count);
    CheckCuda(cudaMemcpy(result.data(), device_c.data(), bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy C to host");
    return result;
}
