#include "block_gemm_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <thread>

namespace {

constexpr int kBlockSize = 32;  // size of matrix block = size of CUDA block

// A single CUDA block computes a single block of C,
// a single CUDA thread computes a single element of C.
__global__ void BlockGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                int n) {
    __shared__ float block_a[kBlockSize][kBlockSize];
    __shared__ float block_b[kBlockSize][kBlockSize];
    __shared__ float block_c[kBlockSize][kBlockSize];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * kBlockSize + ty;
    const int col = blockIdx.x * kBlockSize + tx;

    block_c[ty][tx] = 0.0f;

    for (int k0 = 0; k0 < n; k0 += kBlockSize) {
        // load A block and B block into shared memory
        // (neighbouring threads of a warp read neighbouring addresses)
        const int a_col = k0 + tx;
        const int b_row = k0 + ty;
        block_a[ty][tx] = (row < n && a_col < n) ? a[static_cast<size_t>(row) * n + a_col] : 0.0f;
        block_b[ty][tx] = (b_row < n && col < n) ? b[static_cast<size_t>(b_row) * n + col] : 0.0f;
        __syncthreads();

        // compute BlockA * BlockB and accumulate into C block
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < kBlockSize; ++k) {
            sum += block_a[ty][k] * block_b[k][tx];
        }
        block_c[ty][tx] += sum;
        __syncthreads();
    }

    // dump C block from shared to global memory
    if (row < n && col < n) {
        c[static_cast<size_t>(row) * n + col] = block_c[ty][tx];
    }
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

    const dim3 block(kBlockSize, kBlockSize);
    const dim3 grid((n + kBlockSize - 1) / kBlockSize, (n + kBlockSize - 1) / kBlockSize);
    BlockGemmKernel<<<grid, block>>>(d_a, d_b, d_c, n);

    allocator.join();
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return c;
}
