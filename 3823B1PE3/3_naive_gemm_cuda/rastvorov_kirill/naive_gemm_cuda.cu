#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

namespace {

constexpr int kRowsPerThread = 8;  // each thread computes an 8x4 piece of C
constexpr int kBlockX = 32;        // one warp along the columns
constexpr int kBlockY = 4;

// No shared memory: every value is loaded from global memory directly.
// A thread computes kRowsPerThread rows x 4 columns of C, so each loaded
// element of a and b is reused several times from registers.
// threadIdx.x walks along columns: a warp reads b and writes c as contiguous
// float4, while all threads of the warp read the same a values (broadcast).
// Requires n % 32 == 0, so that all float4 accesses are aligned.
__global__ void NaiveGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c, int n) {
    const int col4 = blockIdx.x * blockDim.x + threadIdx.x;
    const int row0 = (blockIdx.y * blockDim.y + threadIdx.y) * kRowsPerThread;
    if (row0 >= n || col4 * 4 >= n) return;

    const int n4 = n / 4;
    const float4* b4 = reinterpret_cast<const float4*>(b) + col4;

    float4 sum[kRowsPerThread];
#pragma unroll
    for (int r = 0; r < kRowsPerThread; ++r) sum[r] = make_float4(0, 0, 0, 0);

    for (int k = 0; k < n; k += 4) {
        // Four rows k..k+3 of b, 4 columns each
        float4 bv[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) bv[i] = b4[static_cast<size_t>(k + i) * n4];

#pragma unroll
        for (int r = 0; r < kRowsPerThread; ++r) {
            // a[row][k..k+3] in one load
            const float4 av = *reinterpret_cast<const float4*>(
                a + static_cast<size_t>(row0 + r) * n + k);
            const float as[4] = {av.x, av.y, av.z, av.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                sum[r].x += as[i] * bv[i].x;
                sum[r].y += as[i] * bv[i].y;
                sum[r].z += as[i] * bv[i].z;
                sum[r].w += as[i] * bv[i].w;
            }
        }
    }

#pragma unroll
    for (int r = 0; r < kRowsPerThread; ++r) {
        reinterpret_cast<float4*>(c + static_cast<size_t>(row0 + r) * n)[col4] =
            sum[r];
    }
}

// Simple version for small matrices (n < 32): one element per thread
__global__ void NaiveGemmSmallKernel(const float* __restrict__ a,
                                     const float* __restrict__ b,
                                     float* __restrict__ c, int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n || col >= n) return;

    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
        sum += a[row * n + k] * b[k * n + col];
    }
    c[row * n + col] = sum;
}

float* g_buffer = nullptr;
size_t g_capacity = 0;

void EnsureCapacity(size_t elements) {
    if (elements <= g_capacity) return;
    if (g_buffer) cudaFree(g_buffer);
    cudaMalloc(&g_buffer, 3 * elements * sizeof(float));
    g_capacity = elements;
}

}  // namespace

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) return {};
    const size_t elements = static_cast<size_t>(n) * n;
    const size_t bytes = elements * sizeof(float);

    EnsureCapacity(elements);
    float* d_a = g_buffer;
    float* d_b = g_buffer + elements;
    float* d_c = g_buffer + 2 * elements;

    cudaMemcpyAsync(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    if (n % 32 == 0) {
        const dim3 block(kBlockX, kBlockY);
        const dim3 grid((n / 4 + kBlockX - 1) / kBlockX,
                        (n / kRowsPerThread + kBlockY - 1) / kBlockY);
        NaiveGemmKernel<<<grid, block>>>(d_a, d_b, d_c, n);
    } else {
        const dim3 block(16, 16);
        const dim3 grid((n + 15) / 16, (n + 15) / 16);
        NaiveGemmSmallKernel<<<grid, block>>>(d_a, d_b, d_c, n);
    }

    // Host allocation overlaps with the kernel execution
    std::vector<float> c(elements);
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    return c;
}
