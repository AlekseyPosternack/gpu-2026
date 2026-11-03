#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <thread>

namespace {

constexpr int kBlockX = 32;      // one warp along a row of C
constexpr int kBlockY = 16;
constexpr int kRowsPerThread = 8;

// Each thread computes kRowsPerThread elements of one column of C.
// Threads of a warp take neighbouring columns, so loads of B and stores of C
// are coalesced, while loads of A are the same address for the whole warp.
__global__ void NaiveGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = (blockIdx.y * blockDim.y + threadIdx.y) * kRowsPerThread;
    if (col >= n || row >= n) {
        return;
    }

    float acc[kRowsPerThread] = {};
    if (n % 4 == 0) {
        for (int k = 0; k < n; k += 4) {
            const float b0 = b[static_cast<size_t>(k) * n + col];
            const float b1 = b[static_cast<size_t>(k + 1) * n + col];
            const float b2 = b[static_cast<size_t>(k + 2) * n + col];
            const float b3 = b[static_cast<size_t>(k + 3) * n + col];
#pragma unroll
            for (int r = 0; r < kRowsPerThread; ++r) {
                if (row + r < n) {
                    const float4 av = *reinterpret_cast<const float4*>(
                        a + static_cast<size_t>(row + r) * n + k);
                    acc[r] += av.x * b0 + av.y * b1 + av.z * b2 + av.w * b3;
                }
            }
        }
    } else {
        for (int k = 0; k < n; ++k) {
            const float bk = b[static_cast<size_t>(k) * n + col];
#pragma unroll
            for (int r = 0; r < kRowsPerThread; ++r) {
                if (row + r < n) {
                    acc[r] += a[static_cast<size_t>(row + r) * n + k] * bk;
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < kRowsPerThread; ++r) {
        if (row + r < n) {
            c[static_cast<size_t>(row + r) * n + col] = acc[r];
        }
    }
}

}  // namespace

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) {
        return {};
    }
    const size_t count = static_cast<size_t>(n) * n;
    const size_t bytes = count * sizeof(float);

    // allocate result on the host while the device is busy
    std::vector<float> c;
    std::thread allocator([&c, count] { c = std::vector<float>(count); });

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    const dim3 block(kBlockX, kBlockY);
    const int rows_per_block = kBlockY * kRowsPerThread;
    const dim3 grid((n + kBlockX - 1) / kBlockX, (n + rows_per_block - 1) / rows_per_block);
    NaiveGemmKernel<<<grid, block>>>(d_a, d_b, d_c, n);

    allocator.join();
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return c;
}
