#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

namespace {

constexpr int kBlockX = 32;
constexpr int kBlockY = 8;
constexpr int kRowsPerThread = 4;

__global__ void NaiveGemmKernel(const float* __restrict__ a, const float* __restrict__ b,
                                float* __restrict__ c, int n) {
    const int col = blockIdx.x * kBlockX + threadIdx.x;
    const int row0 = (blockIdx.y * kBlockY + threadIdx.y) * kRowsPerThread;
    if (col >= n || row0 >= n) {
        return;
    }

    float acc[kRowsPerThread] = {0.0f};
    const float* a_row = a + static_cast<size_t>(row0) * n;

#pragma unroll 4
    for (int k = 0; k < n; ++k) {
        const float b_val = __ldg(b + static_cast<size_t>(k) * n + col);
#pragma unroll
        for (int r = 0; r < kRowsPerThread; ++r) {
            acc[r] += __ldg(a_row + static_cast<size_t>(r) * n + k) * b_val;
        }
    }

#pragma unroll
    for (int r = 0; r < kRowsPerThread; ++r) {
        if (row0 + r < n) {
            c[static_cast<size_t>(row0 + r) * n + col] = acc[r];
        }
    }
}

__global__ void NaiveGemmSmallKernel(const float* __restrict__ a, const float* __restrict__ b,
                                     float* __restrict__ c, int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n || col >= n) {
        return;
    }
    float acc = 0.0f;
    for (int k = 0; k < n; ++k) {
        acc += a[row * n + k] * b[k * n + col];
    }
    c[row * n + col] = acc;
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

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    cudaMallocAsync(&d_a, bytes, stream);
    cudaMallocAsync(&d_b, bytes, stream);
    cudaMallocAsync(&d_c, bytes, stream);

    cudaMemcpyAsync(d_a, a.data(), bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, b.data(), bytes, cudaMemcpyHostToDevice, stream);

    if (n % (kBlockY * kRowsPerThread) == 0 && n % kBlockX == 0) {
        dim3 block(kBlockX, kBlockY);
        dim3 grid(n / kBlockX, n / (kBlockY * kRowsPerThread));
        NaiveGemmKernel<<<grid, block, 0, stream>>>(d_a, d_b, d_c, n);
    } else {
        dim3 block(16, 16);
        dim3 grid((n + 15) / 16, (n + 15) / 16);
        NaiveGemmSmallKernel<<<grid, block, 0, stream>>>(d_a, d_b, d_c, n);
    }

    std::vector<float> c(count);
    cudaMemcpyAsync(c.data(), d_c, bytes, cudaMemcpyDeviceToHost, stream);

    cudaFreeAsync(d_a, stream);
    cudaFreeAsync(d_b, stream);
    cudaFreeAsync(d_c, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    return c;
}
