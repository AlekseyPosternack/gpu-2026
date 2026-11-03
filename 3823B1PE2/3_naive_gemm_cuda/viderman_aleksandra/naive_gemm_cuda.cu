#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <thread>
#include <vector>

namespace {

constexpr int kBlockX = 32;
constexpr int kBlockY = 4;
constexpr int kRowsPerThread = 8;
constexpr int kColsPerThread = 4;
constexpr int kBlockCols = kBlockX * kColsPerThread;
constexpr int kBlockRows = kBlockY * kRowsPerThread;
constexpr int kSmallBlock = 16;

__device__ __forceinline__ void Fma4(float4& acc, float a, const float4& b) {
    acc.x = fmaf(a, b.x, acc.x);
    acc.y = fmaf(a, b.y, acc.y);
    acc.z = fmaf(a, b.z, acc.z);
    acc.w = fmaf(a, b.w, acc.w);
}

__global__ void NaiveGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                int n) {
    const int col0 = (blockIdx.x * kBlockX + threadIdx.x) * kColsPerThread;
    const int row0 = (blockIdx.y * kBlockY + threadIdx.y) * kRowsPerThread;

    float4 acc[kRowsPerThread] = {};

    for (int k = 0; k < n; k += 4) {
        float4 bv[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            bv[q] = *reinterpret_cast<const float4*>(b + (k + q) * n + col0);
        }

#pragma unroll
        for (int r = 0; r < kRowsPerThread; ++r) {
            const float4 av = *reinterpret_cast<const float4*>(a + (row0 + r) * n + k);
            Fma4(acc[r], av.x, bv[0]);
            Fma4(acc[r], av.y, bv[1]);
            Fma4(acc[r], av.z, bv[2]);
            Fma4(acc[r], av.w, bv[3]);
        }
    }

#pragma unroll
    for (int r = 0; r < kRowsPerThread; ++r) {
        *reinterpret_cast<float4*>(c + (row0 + r) * n + col0) = acc[r];
    }
}

__global__ void SmallGemmKernel(const float* a, const float* b, float* c, int n) {
    const int col = blockIdx.x * kSmallBlock + threadIdx.x;
    const int row = blockIdx.y * kSmallBlock + threadIdx.y;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum = fmaf(a[row * n + k], b[k * n + col], sum);
        }
        c[row * n + col] = sum;
    }
}

struct DeviceMemory {
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
    std::size_t capacity = 0;

    void Ensure(std::size_t count) {
        if (count > capacity) {
            if (a != nullptr) cudaFree(a);
            if (b != nullptr) cudaFree(b);
            if (c != nullptr) cudaFree(c);

            cudaMalloc(&a, count * sizeof(float));
            cudaMalloc(&b, count * sizeof(float));
            cudaMalloc(&c, count * sizeof(float));
            capacity = count;
        }
    }

    ~DeviceMemory() {
        if (a != nullptr) cudaFree(a);
        if (b != nullptr) cudaFree(b);
        if (c != nullptr) cudaFree(c);
    }
};

DeviceMemory g_dev_mem;

}  // namespace

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) {
        return {};
    }

    const std::size_t count = static_cast<std::size_t>(n) * n;
    const std::size_t bytes = count * sizeof(float);

    g_dev_mem.Ensure(count);

    std::vector<float> c;
    std::thread host_alloc([&c, count] { c.resize(count); });

    cudaMemcpy(g_dev_mem.a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(g_dev_mem.b, b.data(), bytes, cudaMemcpyHostToDevice);

    if (n >= kBlockCols && (n % kBlockCols == 0) && (n % kBlockRows == 0)) {
        const dim3 block(kBlockX, kBlockY);
        const dim3 grid(n / kBlockCols, n / kBlockRows);
        NaiveGemmKernel<<<grid, block>>>(g_dev_mem.a, g_dev_mem.b, g_dev_mem.c, n);
    } else {
        const int blocks = (n + kSmallBlock - 1) / kSmallBlock;
        SmallGemmKernel<<<dim3(blocks, blocks), dim3(kSmallBlock, kSmallBlock)>>>(
            g_dev_mem.a, g_dev_mem.b, g_dev_mem.c, n);
    }

    host_alloc.join();

    cudaMemcpy(c.data(), g_dev_mem.c, bytes, cudaMemcpyDeviceToHost);

    return c;
}
