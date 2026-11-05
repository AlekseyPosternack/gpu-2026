#include "block_gemm_cuda.h"

#include <cuda_runtime.h>

namespace {

constexpr int kTile = 64;     // C block computed by one CUDA block: kTile x kTile
constexpr int kTileK = 16;    // depth of A/B blocks loaded into shared memory per step
constexpr int kThreads = 16;  // 16 x 16 threads, each computes 4 x 4 elements of C
constexpr int kPerThread = kTile / kThreads;

// One CUDA block computes a 64x64 block of C. For each pair of A (64x16) and B (16x64) blocks:
// load both into shared memory, synchronize, multiply-accumulate, synchronize.
// Thread (tx, ty) owns elements (ty + 16*i, tx + 16*j), so warp accesses to shared memory
// are either broadcasts or sequential (no bank conflicts) and stores to C are coalesced.
__global__ void BlockGemmKernel(const float* __restrict__ a, const float* __restrict__ b,
                                float* __restrict__ c, int n) {
    __shared__ __align__(16) float block_a[kTile][kTileK];
    __shared__ __align__(16) float block_b[kTileK][kTile];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * kThreads + tx;
    const int row_base = blockIdx.y * kTile;
    const int col_base = blockIdx.x * kTile;

    // Each thread loads one float4 of A block and one float4 of B block
    const int a_row = tid / (kTileK / 4);
    const int a_col = (tid % (kTileK / 4)) * 4;
    const int b_row = tid / (kTile / 4);
    const int b_col = (tid % (kTile / 4)) * 4;
    const float* a_ptr = a + (row_base + a_row) * n + a_col;
    const float* b_ptr = b + b_row * n + col_base + b_col;

    float acc[kPerThread][kPerThread] = {};
    float a_reg[kPerThread];
    float b_reg[kPerThread];

    for (int k0 = 0; k0 < n; k0 += kTileK) {
        *reinterpret_cast<float4*>(&block_a[a_row][a_col]) =
            *reinterpret_cast<const float4*>(a_ptr + k0);
        *reinterpret_cast<float4*>(&block_b[b_row][b_col]) =
            *reinterpret_cast<const float4*>(b_ptr + k0 * n);
        __syncthreads();

#pragma unroll
        for (int k = 0; k < kTileK; ++k) {
#pragma unroll
            for (int i = 0; i < kPerThread; ++i) a_reg[i] = block_a[ty + i * kThreads][k];
#pragma unroll
            for (int j = 0; j < kPerThread; ++j) b_reg[j] = block_b[k][tx + j * kThreads];
#pragma unroll
            for (int i = 0; i < kPerThread; ++i)
#pragma unroll
                for (int j = 0; j < kPerThread; ++j) acc[i][j] += a_reg[i] * b_reg[j];
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < kPerThread; ++i)
#pragma unroll
        for (int j = 0; j < kPerThread; ++j)
            c[(row_base + ty + i * kThreads) * n + col_base + tx + j * kThreads] = acc[i][j];
}

// Classic one-element-per-thread block version, used for small matrices (n < 64)
constexpr int kSmallTile = 16;

__global__ void SmallBlockGemmKernel(const float* a, const float* b, float* c, int n) {
    __shared__ float block_a[kSmallTile][kSmallTile];
    __shared__ float block_b[kSmallTile][kSmallTile];
    const int row = blockIdx.y * kSmallTile + threadIdx.y;
    const int col = blockIdx.x * kSmallTile + threadIdx.x;
    float sum = 0.0f;
    for (int k0 = 0; k0 < n; k0 += kSmallTile) {
        const int ka = k0 + threadIdx.x;
        const int kb = k0 + threadIdx.y;
        block_a[threadIdx.y][threadIdx.x] = (row < n && ka < n) ? a[row * n + ka] : 0.0f;
        block_b[threadIdx.y][threadIdx.x] = (kb < n && col < n) ? b[kb * n + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < kSmallTile; ++k) sum += block_a[threadIdx.y][k] * block_b[k][threadIdx.x];
        __syncthreads();
    }
    if (row < n && col < n) c[row * n + col] = sum;
}

struct Buffers {
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
    size_t capacity = 0;

    void Reserve(size_t count) {
        if (count <= capacity) return;
        cudaFree(a);
        cudaFree(b);
        cudaFree(c);
        cudaMalloc(&a, count * sizeof(float));
        cudaMalloc(&b, count * sizeof(float));
        cudaMalloc(&c, count * sizeof(float));
        capacity = count;
    }
};

}  // namespace

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) return {};
    const size_t count = size_t(n) * n;
    const size_t bytes = count * sizeof(float);

    static Buffers buf;  // device memory is allocated once and reused between calls
    buf.Reserve(count);

    cudaMemcpy(buf.a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(buf.b, b.data(), bytes, cudaMemcpyHostToDevice);

    if (n % kTile == 0) {
        BlockGemmKernel<<<dim3(n / kTile, n / kTile), dim3(kThreads, kThreads)>>>(
            buf.a, buf.b, buf.c, n);
    } else {
        const int blocks = (n + kSmallTile - 1) / kSmallTile;
        SmallBlockGemmKernel<<<dim3(blocks, blocks), dim3(kSmallTile, kSmallTile)>>>(
            buf.a, buf.b, buf.c, n);
    }

    // Kernel launch is asynchronous, so the result is allocated on host while it runs
    std::vector<float> c(count);
    cudaMemcpy(c.data(), buf.c, bytes, cudaMemcpyDeviceToHost);
    return c;
}
