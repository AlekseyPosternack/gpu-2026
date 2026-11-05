#include "block_gemm_cuda.h"

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

constexpr int kTile = 32;
constexpr int kBlockX = 32;
constexpr int kBlockY = 8;
constexpr int kRowsPerThread = kTile / kBlockY;

__global__ void BlockGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                int n) {
    __shared__ float sa[kTile][kTile];
    __shared__ float sb[kTile][kTile];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int col = blockIdx.x * kTile + tx;
    const int row0 = blockIdx.y * kTile + ty;

    float acc[kRowsPerThread] = {};

    for (int t = 0; t < n; t += kTile) {
#pragma unroll
        for (int r = 0; r < kRowsPerThread; ++r) {
            const int i = ty + r * kBlockY;
            const int a_row = blockIdx.y * kTile + i;
            const int a_col = t + tx;
            const int b_row = t + i;
            sa[i][tx] = (a_row < n && a_col < n) ? a[size_t(a_row) * n + a_col] : 0.0f;
            sb[i][tx] = (b_row < n && col < n) ? b[size_t(b_row) * n + col] : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < kTile; ++k) {
            const float bv = sb[k][tx];
#pragma unroll
            for (int r = 0; r < kRowsPerThread; ++r) {
                acc[r] += sa[ty + r * kBlockY][k] * bv;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < kRowsPerThread; ++r) {
        const int row = row0 + r * kBlockY;
        if (row < n && col < n) {
            c[size_t(row) * n + col] = acc[r];
        }
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

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
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

    const dim3 block(kBlockX, kBlockY);
    const dim3 grid((n + kTile - 1) / kTile, (n + kTile - 1) / kTile);
    BlockGemmKernel<<<grid, block>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());

    alloc_thread.join();

    CUDA_CHECK(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    return c;
}