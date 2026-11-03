#include "block_gemm_cuda.h"

#include <cuda_runtime.h>

namespace {

constexpr int kTile = 32;
constexpr int kRowsPerThread = 4;
constexpr int kThreadsY = kTile / kRowsPerThread;

__global__ void BlockGemmKernel(const float* __restrict__ a, const float* __restrict__ b,
                                float* __restrict__ c, int n) {
    __shared__ float tile_a[kTile][kTile];
    __shared__ float tile_b[kTile][kTile];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int col = blockIdx.x * kTile + tx;
    const int row_base = blockIdx.y * kTile;

    float acc[kRowsPerThread] = {0.0f};

    for (int t = 0; t < n; t += kTile) {
#pragma unroll
        for (int r = 0; r < kRowsPerThread; ++r) {
            const int local_row = ty + r * kThreadsY;
            tile_a[local_row][tx] = a[static_cast<size_t>(row_base + local_row) * n + t + tx];
            tile_b[local_row][tx] = b[static_cast<size_t>(t + local_row) * n + col];
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < kTile; ++k) {
            const float b_val = tile_b[k][tx];
#pragma unroll
            for (int r = 0; r < kRowsPerThread; ++r) {
                acc[r] += tile_a[ty + r * kThreadsY][k] * b_val;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < kRowsPerThread; ++r) {
        c[static_cast<size_t>(row_base + ty + r * kThreadsY) * n + col] = acc[r];
    }
}

__global__ void SmallGemmKernel(const float* __restrict__ a, const float* __restrict__ b,
                                float* __restrict__ c, int n) {
    extern __shared__ float smem[];
    float* sa = smem;
    float* sb = smem + n * n;
    const int col = threadIdx.x;
    const int row = threadIdx.y;
    sa[row * n + col] = a[row * n + col];
    sb[row * n + col] = b[row * n + col];
    __syncthreads();
    float acc = 0.0f;
    for (int k = 0; k < n; ++k) {
        acc += sa[row * n + k] * sb[k * n + col];
    }
    c[row * n + col] = acc;
}

}  // namespace

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
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

    if (n >= kTile) {
        dim3 block(kTile, kThreadsY);
        dim3 grid(n / kTile, n / kTile);
        BlockGemmKernel<<<grid, block, 0, stream>>>(d_a, d_b, d_c, n);
    } else {
        dim3 block(n, n);
        SmallGemmKernel<<<1, block, 2 * count * sizeof(float), stream>>>(d_a, d_b, d_c, n);
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
