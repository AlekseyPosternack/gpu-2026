#include "block_gemm_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace {

constexpr int kTileSize = 16;

void CheckCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

__global__ void BlockGemmKernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n) {
    __shared__ float tileA[kTileSize][kTileSize];
    __shared__ float tileB[kTileSize][kTileSize];

    const int column = blockIdx.x * kTileSize + threadIdx.x;
    const int row = blockIdx.y * kTileSize + threadIdx.y;
    float sum = 0.0f;

    for (int tile = 0; tile < (n + kTileSize - 1) / kTileSize; ++tile) {
        const int aColumn = tile * kTileSize + threadIdx.x;
        const int bRow = tile * kTileSize + threadIdx.y;
        tileA[threadIdx.y][threadIdx.x] = row < n && aColumn < n ? a[row * n + aColumn] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = bRow < n && column < n ? b[bRow * n + column] : 0.0f;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < kTileSize; ++k) {
            sum = fmaf(tileA[threadIdx.y][k], tileB[k][threadIdx.x], sum);
        }
        __syncthreads();
    }

    if (row < n && column < n) {
        c[row * n + column] = sum;
    }
}

class DeviceStorage {
public:
    ~DeviceStorage() {
        cudaFree(data_);
    }

    float* Reserve(std::size_t size) {
        if (size <= capacity_) {
            return data_;
        }
        if (data_ != nullptr) {
            CheckCuda(cudaFree(data_), "cudaFree");
            data_ = nullptr;
            capacity_ = 0;
        }
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&data_), size * sizeof(float)), "cudaMalloc");
        capacity_ = size;
        return data_;
    }

private:
    float* data_ = nullptr;
    std::size_t capacity_ = 0;
};

}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    if (n < 0) {
        throw std::invalid_argument("matrix size must not be negative");
    }
    const std::size_t count = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    if (a.size() != count || b.size() != count) {
        throw std::invalid_argument("matrix dimensions do not match n");
    }
    if (count == 0) {
        return {};
    }

    const std::size_t bytes = count * sizeof(float);
    static thread_local DeviceStorage storage;
    float* deviceA = storage.Reserve(count * 3);
    float* deviceB = deviceA + count;
    float* deviceC = deviceB + count;

    CheckCuda(cudaMemcpy(deviceA, a.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy a");
    CheckCuda(cudaMemcpy(deviceB, b.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy b");

    const dim3 block(kTileSize, kTileSize);
    const dim3 grid((n + kTileSize - 1) / kTileSize, (n + kTileSize - 1) / kTileSize);
    BlockGemmKernel<<<grid, block>>>(deviceA, deviceB, deviceC, n);
    CheckCuda(cudaGetLastError(), "BlockGemmKernel launch");

    std::vector<float> result(count);
    CheckCuda(cudaMemcpy(result.data(), deviceC, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy result");
    return result;
}
