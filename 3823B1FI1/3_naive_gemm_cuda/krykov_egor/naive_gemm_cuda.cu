#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <thread>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                        \
            std::fprintf(stderr, "CUDA error \"%s\" at %s:%d\n",          \
                         cudaGetErrorString(err_), __FILE__, __LINE__);   \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

namespace {

constexpr int kBlockX = 32;
constexpr int kBlockY = 8;
constexpr int kRowsPerThread = 4;
constexpr int kColsPerThread = 4;
constexpr int kTileRows = kBlockY * kRowsPerThread;
constexpr int kTileCols = kBlockX * kColsPerThread;
constexpr int kSmallBlock = 16;

__device__ __forceinline__ float4 Load4(const float* p) {
    return *reinterpret_cast<const float4*>(p);
}


__device__ __forceinline__ void Fma4(float4& acc, float a, const float4& b) {
    acc.x = fmaf(a, b.x, acc.x);
    acc.y = fmaf(a, b.y, acc.y);
    acc.z = fmaf(a, b.z, acc.z);
    acc.w = fmaf(a, b.w, acc.w);
}

__global__ void GemmKernel(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ c,
                           int n) {
    const int col = (blockIdx.x * kBlockX + threadIdx.x) * kColsPerThread;
    const int row0 = blockIdx.y * kTileRows + threadIdx.y;

    float4 acc[kRowsPerThread] = {};

    for (int k = 0; k < n; k += 4) {
        float4 bv[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            bv[q] = Load4(b + size_t(k + q) * n + col);
        }
#pragma unroll
        for (int r = 0; r < kRowsPerThread; ++r) {
            const float4 av = Load4(a + size_t(row0 + r * kBlockY) * n + k);
            Fma4(acc[r], av.x, bv[0]);
            Fma4(acc[r], av.y, bv[1]);
            Fma4(acc[r], av.z, bv[2]);
            Fma4(acc[r], av.w, bv[3]);
        }
    }

#pragma unroll
    for (int r = 0; r < kRowsPerThread; ++r) {
        *reinterpret_cast<float4*>(c + size_t(row0 + r * kBlockY) * n + col) = acc[r];
    }
}


__global__ void SmallGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                int n) {
    const int col = blockIdx.x * kSmallBlock + threadIdx.x;
    const int row = blockIdx.y * kSmallBlock + threadIdx.y;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum = fmaf(a[size_t(row) * n + k], b[size_t(k) * n + col], sum);
        }
        c[size_t(row) * n + col] = sum;
    }
}

struct DeviceBuffer {
    float* ptr = nullptr;
    size_t capacity = 0;

    float* Get(size_t count) {
        if (count > capacity) {
            if (ptr != nullptr) {
                CUDA_CHECK(cudaFree(ptr));
            }
            CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(float)));
            capacity = count;
        }
        return ptr;
    }

    ~DeviceBuffer() {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }
};

DeviceBuffer g_a, g_b, g_c;

}  // namespace

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) {
        return {};
    }
    const size_t count = static_cast<size_t>(n) * n;
    const size_t bytes = count * sizeof(float);

    float* d_a = g_a.Get(count);
    float* d_b = g_b.Get(count);
    float* d_c = g_c.Get(count);

    std::vector<float> c;
    std::thread alloc_thread([&c, count] { c.resize(count); });

    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));

    if (n < kTileCols) {
        const int blocks = (n + kSmallBlock - 1) / kSmallBlock;
        SmallGemmKernel<<<dim3(blocks, blocks), dim3(kSmallBlock, kSmallBlock)>>>(d_a, d_b, d_c, n);
    } else {
        const dim3 block(kBlockX, kBlockY);
        const dim3 grid(n / kTileCols, n / kTileRows);
        GemmKernel<<<grid, block>>>(d_a, d_b, d_c, n);
    }
    CUDA_CHECK(cudaGetLastError());

    alloc_thread.join();
    CUDA_CHECK(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    return c;
}