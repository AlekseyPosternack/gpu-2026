#include "block_gemm_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>

namespace {

constexpr int tile_size = 32;

void checkCuda(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

__global__ void blockMultiplyKernel(const float* a, const float* b, float* c, int n) {
    __shared__ float tile_a[tile_size][tile_size];
    __shared__ float tile_b[tile_size][tile_size];
    __shared__ float tile_c[tile_size][tile_size];

    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int row = blockIdx.y * tile_size + y;
    const int col = blockIdx.x * tile_size + x;
    tile_c[y][x] = 0.0f;

    const int tile_count = (n - 1) / tile_size + 1;
    for (int tile = 0; tile < tile_count; ++tile) {
        const int a_col = tile * tile_size + x;
        const int b_row = tile * tile_size + y;
        tile_a[y][x] = row < n && a_col < n
            ? a[static_cast<std::size_t>(row) * n + a_col] : 0.0f;
        tile_b[y][x] = b_row < n && col < n
            ? b[static_cast<std::size_t>(b_row) * n + col] : 0.0f;
        __syncthreads();

        float sum = tile_c[y][x];
        #pragma unroll
        for (int k = 0; k < tile_size; ++k) {
            sum += tile_a[y][k] * tile_b[k][x];
        }
        tile_c[y][x] = sum;
        __syncthreads();
    }

    if (row < n && col < n) {
        c[static_cast<std::size_t>(row) * n + col] = tile_c[y][x];
    }
}

}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                               const std::vector<float>& b,
                               int n) {
    if (n < 0) {
        throw std::invalid_argument("Matrix size must be nonnegative");
    }
    const std::size_t size = static_cast<std::size_t>(n) * n;
    if (a.size() != size || b.size() != size) {
        throw std::invalid_argument("Matrix dimensions do not match the data");
    }
    if (n == 0) {
        return {};
    }

    const std::size_t bytes = size * sizeof(float);
    float* device_data = nullptr;
    checkCuda(cudaMalloc(&device_data, 3 * bytes));
    float* device_a = device_data;
    float* device_b = device_data + size;
    float* device_c = device_data + 2 * size;

    std::vector<float> result;
    try {
        checkCuda(cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice));

        const dim3 threads(tile_size, tile_size);
        const unsigned int count = (n - 1) / tile_size + 1;
        const dim3 blocks(count, count);
        blockMultiplyKernel<<<blocks, threads>>>(device_a, device_b, device_c, n);
        checkCuda(cudaGetLastError());

        result.resize(size);
        checkCuda(cudaMemcpy(result.data(), device_c, bytes, cudaMemcpyDeviceToHost));
    } catch (...) {
        cudaFree(device_data);
        throw;
    }

    checkCuda(cudaFree(device_data));
    return result;
}
