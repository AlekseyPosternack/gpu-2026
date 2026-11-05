#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <thread>

namespace {

// Each thread computes a TILE x TILE block of C held entirely in registers.
// Loading a whole float4 of A and of B per k-step with 4x4 reuse turns the
// memory-bound naive algorithm into an FMA-bound one: per k-step a thread does
// 4 + 4 vector loads and TILE * TILE * TILE = 64 FMAs.
constexpr int kTile = 4;
constexpr int kThreads = 16;                      // 16 x 16 threads per block
constexpr int kBlockTile = kTile * kThreads;      // 64 x 64 block tile of C

__global__ void NaiveGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c, int n) {
    const int row0 = (blockIdx.y * blockDim.y + threadIdx.y) * kTile;
    const int col0 = (blockIdx.x * blockDim.x + threadIdx.x) * kTile;

    float acc[kTile][kTile] = {};

    for (int k = 0; k < n; k += kTile) {
        float4 av[kTile];
#pragma unroll
        for (int r = 0; r < kTile; ++r) {
            av[r] = *reinterpret_cast<const float4*>(a + (row0 + r) * n + k);
        }
        float4 bv[kTile];
#pragma unroll
        for (int q = 0; q < kTile; ++q) {
            bv[q] = *reinterpret_cast<const float4*>(b + (k + q) * n + col0);
        }
#pragma unroll
        for (int r = 0; r < kTile; ++r) {
            const float ark[kTile] = {av[r].x, av[r].y, av[r].z, av[r].w};
#pragma unroll
            for (int q = 0; q < kTile; ++q) {
                const float bkc[kTile] = {bv[q].x, bv[q].y, bv[q].z, bv[q].w};
#pragma unroll
                for (int cq = 0; cq < kTile; ++cq) {
                    acc[r][cq] = fmaf(ark[q], bkc[cq], acc[r][cq]);
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < kTile; ++r) {
        *reinterpret_cast<float4*>(c + (row0 + r) * n + col0) =
            make_float4(acc[r][0], acc[r][1], acc[r][2], acc[r][3]);
    }
}

// Fallback: one element per thread, no alignment assumptions.
__global__ void GenericGemmKernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c, int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n || col >= n) {
        return;
    }
    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
        sum += a[row * n + k] * b[k * n + col];
    }
    c[row * n + col] = sum;
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

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
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

    if (n % kBlockTile == 0) {
        const dim3 block(kThreads, kThreads);
        const dim3 grid(n / kBlockTile, n / kBlockTile);
        NaiveGemmKernel<<<grid, block>>>(da, db, dc, n);
    } else {
        constexpr int kFallback = 16;
        const int blocks = (n + kFallback - 1) / kFallback;
        GenericGemmKernel<<<dim3(blocks, blocks), dim3(kFallback, kFallback)>>>(da, db, dc, n);
    }

    allocator.join();
    cudaMemcpy(result.data(), dc, bytes, cudaMemcpyDeviceToHost);
    return result;
}
