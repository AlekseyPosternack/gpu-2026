#include "block_gemm_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <thread>

namespace {

constexpr int kTile = 64;     // C block computed by one CUDA block: kTile x kTile
constexpr int kTileK = 16;    // depth of A/B blocks staged in shared memory per step
constexpr int kThreads = 16;  // 16 x 16 threads, each computes 4 x 4 elements of C
constexpr int kPerThread = kTile / kThreads;

// One CUDA block computes a kTile x kTile block of C. For each pair of A (kTile x kTileK)
// and B (kTileK x kTile) blocks: load both into shared memory, synchronize,
// multiply-accumulate, synchronize. Thread (tx, ty) owns elements
// (ty + 16*i, tx + 16*j), so shared-memory reads are either broadcasts or
// sequential and stores to C are coalesced.
__global__ void BlockGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c, int n) {
    __shared__ __align__(16) float block_a[kTile][kTileK];
    __shared__ __align__(16) float block_b[kTileK][kTile];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * kThreads + tx;
    const int row_base = blockIdx.y * kTile;
    const int col_base = blockIdx.x * kTile;

    // Each of the 256 threads loads one float4 of A and one float4 of B per step.
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
            for (int i = 0; i < kPerThread; ++i) {
                a_reg[i] = block_a[ty + i * kThreads][k];
            }
#pragma unroll
            for (int j = 0; j < kPerThread; ++j) {
                b_reg[j] = block_b[k][tx + j * kThreads];
            }
#pragma unroll
            for (int i = 0; i < kPerThread; ++i) {
#pragma unroll
                for (int j = 0; j < kPerThread; ++j) {
                    acc[i][j] = fmaf(a_reg[i], b_reg[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < kPerThread; ++i) {
#pragma unroll
        for (int j = 0; j < kPerThread; ++j) {
            c[(row_base + ty + i * kThreads) * n + col_base + tx + j * kThreads] = acc[i][j];
        }
    }
}

// Fallback for matrices whose size is not a multiple of kTile.
constexpr int kSmallTile = 16;

__global__ void SmallBlockGemmKernel(const float* __restrict__ a,
                                     const float* __restrict__ b,
                                     float* __restrict__ c, int n) {
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
        for (int k = 0; k < kSmallTile; ++k) {
            sum += block_a[threadIdx.y][k] * block_b[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

// Device memory is allocated once and reused across calls.
void ReserveDevice(float** a, float** b, float** c, std::size_t count) {
    static float* da = nullptr;
    static float* db = nullptr;
    static float* dc = nullptr;
    static std::size_t capacity = 0;

    if (count > capacity) {
        cudaFree(da);
        cudaFree(db);
        cudaFree(dc);
        cudaMalloc(&da, count * sizeof(float));
        cudaMalloc(&db, count * sizeof(float));
        cudaMalloc(&dc, count * sizeof(float));
        capacity = count;
    }
    *a = da;
    *b = db;
    *c = dc;
}

}  // namespace

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) {
        return {};
    }
    const std::size_t count = static_cast<std::size_t>(n) * n;
    const std::size_t bytes = count * sizeof(float);

    float* da = nullptr;
    float* db = nullptr;
    float* dc = nullptr;
    ReserveDevice(&da, &db, &dc, count);

    // Host allocation (zero-fill + page faults) overlaps with the copies.
    std::vector<float> result;
    std::thread allocator([&result, count] { result.resize(count); });

    cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice);

    if (n % kTile == 0) {
        const dim3 block(kThreads, kThreads);
        const dim3 grid(n / kTile, n / kTile);
        BlockGemmKernel<<<grid, block>>>(da, db, dc, n);
    } else {
        const int blocks = (n + kSmallTile - 1) / kSmallTile;
        SmallBlockGemmKernel<<<dim3(blocks, blocks), dim3(kSmallTile, kSmallTile)>>>(da, db, dc, n);
    }

    allocator.join();
    cudaMemcpy(result.data(), dc, bytes, cudaMemcpyDeviceToHost);
    return result;
}
