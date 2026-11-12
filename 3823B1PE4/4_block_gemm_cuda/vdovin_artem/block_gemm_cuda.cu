#include "block_gemm_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace {

constexpr int kTile = 16;

void Check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

float* g_device = nullptr;
std::size_t g_capacity = 0;

float* GetDeviceBuffer(std::size_t size) {
    if (size > g_capacity) {
        if (g_device != nullptr) {
            cudaFree(g_device);
            g_device = nullptr;
        }
        Check(cudaMalloc(&g_device, size * sizeof(float)), "cudaMalloc");
        g_capacity = size;
    }
    return g_device;
}

__global__ void BlockGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                int n) {
    __shared__ float tile_a[kTile][kTile];
    __shared__ float tile_b[kTile][kTile + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * kTile + ty;
    const int col = blockIdx.x * kTile + tx;

    float sum = 0.0f;
    for (int start = 0; start < n; start += kTile) {
        const int a_col = start + tx;
        const int b_row = start + ty;
        tile_a[ty][tx] = (row < n && a_col < n) ? a[static_cast<std::size_t>(row) * n + a_col] : 0.0f;
        tile_b[ty][tx] = (b_row < n && col < n) ? b[static_cast<std::size_t>(b_row) * n + col] : 0.0f;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < kTile; ++k) {
            sum = fmaf(tile_a[ty][k], tile_b[k][tx], sum);
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        c[static_cast<std::size_t>(row) * n + col] = sum;
    }
}

}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    const std::size_t count = static_cast<std::size_t>(n) * n;
    std::vector<float> c(count);
    if (n <= 0) {
        return c;
    }

    const std::size_t bytes = count * sizeof(float);
    float* d_a = GetDeviceBuffer(3 * count);
    float* d_b = d_a + count;
    float* d_c = d_b + count;

    Check(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy A");
    Check(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy B");

    const dim3 threads(kTile, kTile);
    const dim3 blocks((n + kTile - 1) / kTile, (n + kTile - 1) / kTile);
    BlockGemmKernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
    Check(cudaGetLastError(), "BlockGemmKernel");

    Check(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy C");
    return c;
}
