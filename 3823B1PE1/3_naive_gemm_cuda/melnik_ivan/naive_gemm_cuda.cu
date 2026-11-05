#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <thread>
#include <vector>

namespace {

constexpr int kBlockThreadsX = 32;
constexpr int kBlockThreadsY = 8;
constexpr int kRowsPerThread = 4;
constexpr int kColsPerThread = 4;

constexpr int kTileCols = kBlockThreadsX * kColsPerThread;  // 128
constexpr int kTileRows = kBlockThreadsY * kRowsPerThread;  // 32
constexpr int kFallbackBlock = 16;

__device__ __forceinline__ void AccumulateFma4(float4& acc, float a_val, const float4& b_vec) {
    acc.x = fmaf(a_val, b_vec.x, acc.x);
    acc.y = fmaf(a_val, b_vec.y, acc.y);
    acc.z = fmaf(a_val, b_vec.z, acc.z);
    acc.w = fmaf(a_val, b_vec.w, acc.w);
}

__global__ void NaiveGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                int n) {
    const int col0 = (blockIdx.x * kBlockThreadsX + threadIdx.x) * kColsPerThread;
    const int row0 = (blockIdx.y * kBlockThreadsY + threadIdx.y) * kRowsPerThread;

    float4 acc[kRowsPerThread];
#pragma unroll
    for (int r = 0; r < kRowsPerThread; ++r) {
        acc[r] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    for (int k = 0; k < n; k += 4) {
        const float4 b0 = *reinterpret_cast<const float4*>(b + static_cast<size_t>(k + 0) * n + col0);
        const float4 b1 = *reinterpret_cast<const float4*>(b + static_cast<size_t>(k + 1) * n + col0);
        const float4 b2 = *reinterpret_cast<const float4*>(b + static_cast<size_t>(k + 2) * n + col0);
        const float4 b3 = *reinterpret_cast<const float4*>(b + static_cast<size_t>(k + 3) * n + col0);

#pragma unroll
        for (int r = 0; r < kRowsPerThread; ++r) {
            const float4 a_vec = *reinterpret_cast<const float4*>(a + static_cast<size_t>(row0 + r) * n + k);
            AccumulateFma4(acc[r], a_vec.x, b0);
            AccumulateFma4(acc[r], a_vec.y, b1);
            AccumulateFma4(acc[r], a_vec.z, b2);
            AccumulateFma4(acc[r], a_vec.w, b3);
        }
    }

#pragma unroll
    for (int r = 0; r < kRowsPerThread; ++r) {
        *reinterpret_cast<float4*>(c + static_cast<size_t>(row0 + r) * n + col0) = acc[r];
    }
}

__global__ void NaiveGemmFallbackKernel(const float* __restrict__ a,
                                        const float* __restrict__ b,
                                        float* __restrict__ c,
                                        int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum = fmaf(a[row * n + k], b[k * n + col], sum);
        }
        c[row * n + col] = sum;
    }
}

struct DeviceGemmBuffer {
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    size_t capacity = 0;
    cudaStream_t stream = nullptr;

    DeviceGemmBuffer() {
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    }

    ~DeviceGemmBuffer() {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
        if (stream) cudaStreamDestroy(stream);
    }

    void EnsureCapacity(size_t count) {
        if (count > capacity) {
            if (d_a) cudaFree(d_a);
            if (d_b) cudaFree(d_b);
            if (d_c) cudaFree(d_c);

            cudaMalloc(&d_a, count * sizeof(float));
            cudaMalloc(&d_b, count * sizeof(float));
            cudaMalloc(&d_c, count * sizeof(float));
            capacity = count;
        }
    }
};

DeviceGemmBuffer& GetDeviceBuffer() {
    static DeviceGemmBuffer buffer;
    return buffer;
}

}  // namespace

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) {
        return {};
    }

    const size_t count = static_cast<size_t>(n) * n;
    if (a.size() < count || b.size() < count) {
        return {};
    }
    const size_t bytes = count * sizeof(float);

    DeviceGemmBuffer& buf = GetDeviceBuffer();
    buf.EnsureCapacity(count);

    // Overlap host memory allocation with asynchronous device transfers and computation
    std::vector<float> c;
    std::thread host_allocator([&c, count] {
        c.resize(count);
    });

    cudaMemcpyAsync(buf.d_a, a.data(), bytes, cudaMemcpyHostToDevice, buf.stream);
    cudaMemcpyAsync(buf.d_b, b.data(), bytes, cudaMemcpyHostToDevice, buf.stream);

    if (n >= kTileCols && (n % kTileCols == 0) && (n % kTileRows == 0) && (n % 4 == 0)) {
        const dim3 block(kBlockThreadsX, kBlockThreadsY);
        const dim3 grid(n / kTileCols, n / kTileRows);
        NaiveGemmKernel<<<grid, block, 0, buf.stream>>>(buf.d_a, buf.d_b, buf.d_c, n);
    } else {
        const int blocks = (n + kFallbackBlock - 1) / kFallbackBlock;
        const dim3 block(kFallbackBlock, kFallbackBlock);
        const dim3 grid(blocks, blocks);
        NaiveGemmFallbackKernel<<<grid, block, 0, buf.stream>>>(buf.d_a, buf.d_b, buf.d_c, n);
    }

    host_allocator.join();

    cudaMemcpyAsync(c.data(), buf.d_c, bytes, cudaMemcpyDeviceToHost, buf.stream);
    cudaStreamSynchronize(buf.stream);

    return c;
}
