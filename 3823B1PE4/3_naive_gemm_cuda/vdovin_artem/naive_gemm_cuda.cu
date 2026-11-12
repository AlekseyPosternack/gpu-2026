#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace {

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

__global__ void NaiveGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) {
        return;
    }

    const float* a_row = a + static_cast<std::size_t>(row) * n;
    float sum = 0.0f;
#pragma unroll 8
    for (int k = 0; k < n; ++k) {
        sum = fmaf(a_row[k], b[static_cast<std::size_t>(k) * n + col], sum);
    }
    c[static_cast<std::size_t>(row) * n + col] = sum;
}

}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
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

    const dim3 threads(32, 8);
    const dim3 blocks((n + threads.x - 1) / threads.x, (n + threads.y - 1) / threads.y);
    NaiveGemmKernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
    Check(cudaGetLastError(), "NaiveGemmKernel");

    Check(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy C");
    return c;
}
