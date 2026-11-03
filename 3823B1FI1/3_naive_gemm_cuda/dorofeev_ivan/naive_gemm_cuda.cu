#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <thread>

namespace {

constexpr int kRowsPerThread = 16;
constexpr int kGroupsPerThread = 2;
constexpr int kWarps = 4;
constexpr int kTileRows = kRowsPerThread * kWarps;
constexpr int kTileCols = 32 * 4 * kGroupsPerThread;
constexpr int kChunks = 8;
constexpr int kParts = 2;
constexpr int kSmallBlock = 16;

__device__ __forceinline__ float4 Load4(const float* p) {
    return *reinterpret_cast<const float4*>(p);
}

__device__ __forceinline__ void Fma4(float4& acc, float a, const float4& b) {
    acc.x = fmaf(a, b.x, acc.x);
    acc.y = fmaf(a, b.y, acc.y);
    acc.z = fmaf(a, b.z, acc.z);
    acc.w = fmaf(a, b.w, acc.w);
}

template <bool kAccumulate>
__global__ void __launch_bounds__(32 * kWarps) GemmKernel(const float* __restrict__ a, const float* __restrict__ b,
                                                         float* __restrict__ c, int n, int k_begin, int k_end) {
    const int row = (blockIdx.x * kWarps + threadIdx.y) * kRowsPerThread;
    const int col = blockIdx.y * kTileCols + threadIdx.x * 4;

    float4 acc[kRowsPerThread][kGroupsPerThread] = {};
    if (kAccumulate) {
#pragma unroll
        for (int i = 0; i < kRowsPerThread; ++i) {
#pragma unroll
            for (int g = 0; g < kGroupsPerThread; ++g) {
                acc[i][g] = Load4(c + size_t(row + i) * n + col + g * 128);
            }
        }
    }
    for (int k = k_begin; k < k_end; k += 4) {
        float4 bv[4][kGroupsPerThread];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
#pragma unroll
            for (int g = 0; g < kGroupsPerThread; ++g) {
                bv[q][g] = Load4(b + size_t(k + q) * n + col + g * 128);
            }
        }
#pragma unroll
        for (int i = 0; i < kRowsPerThread; ++i) {
            const float4 av = Load4(a + size_t(row + i) * n + k);
#pragma unroll
            for (int g = 0; g < kGroupsPerThread; ++g) {
                Fma4(acc[i][g], av.x, bv[0][g]);
                Fma4(acc[i][g], av.y, bv[1][g]);
                Fma4(acc[i][g], av.z, bv[2][g]);
                Fma4(acc[i][g], av.w, bv[3][g]);
            }
        }
    }

#pragma unroll
    for (int i = 0; i < kRowsPerThread; ++i) {
#pragma unroll
        for (int g = 0; g < kGroupsPerThread; ++g) {
            *reinterpret_cast<float4*>(c + size_t(row + i) * n + col + g * 128) = acc[i][g];
        }
    }
}

__global__ void SmallGemmKernel(const float* a, const float* b, float* c, int n) {
    const int row = blockIdx.y * kSmallBlock + threadIdx.y;
    const int col = blockIdx.x * kSmallBlock + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum = fmaf(a[row * n + k], b[k * n + col], sum);
        }
        c[row * n + col] = sum;
    }
}

struct Context {
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
    size_t capacity = 0;
    cudaStream_t streams[kChunks];
    cudaStream_t copy_stream;
    cudaEvent_t b_ready[kParts];

    Context() {
        for (cudaStream_t& s : streams) {
            cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        }
        cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking);
        for (cudaEvent_t& e : b_ready) {
            cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
        }
    }

    ~Context() {
        cudaFree(a);
        cudaFree(b);
        cudaFree(c);
        for (cudaStream_t s : streams) {
            cudaStreamDestroy(s);
        }
        cudaStreamDestroy(copy_stream);
        for (cudaEvent_t e : b_ready) {
            cudaEventDestroy(e);
        }
    }

    void Reserve(size_t count) {
        if (count > capacity) {
            cudaFree(a);
            cudaFree(b);
            cudaFree(c);
            cudaMalloc(&a, count * sizeof(float));
            cudaMalloc(&b, count * sizeof(float));
            cudaMalloc(&c, count * sizeof(float));
            capacity = count;
        }
    }
};

Context& GetContext() {
    static Context context;
    return context;
}

}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    const size_t count = size_t(n) * n;
    const size_t bytes = count * sizeof(float);
    Context& ctx = GetContext();
    ctx.Reserve(count);

    if (n < kTileCols) {
        std::vector<float> c(count);
        if (n > 0) {
            cudaMemcpy(ctx.a, a.data(), bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(ctx.b, b.data(), bytes, cudaMemcpyHostToDevice);
            const int blocks = (n + kSmallBlock - 1) / kSmallBlock;
            SmallGemmKernel<<<dim3(blocks, blocks), dim3(kSmallBlock, kSmallBlock)>>>(ctx.a, ctx.b, ctx.c, n);
            cudaMemcpy(c.data(), ctx.c, bytes, cudaMemcpyDeviceToHost);
        }
        return c;
    }

    std::vector<float> c;
    std::thread builder([&] { c.resize(count); });

    const int rows = std::max(kTileRows, n / kChunks);
    const int chunks = n / rows;
    const size_t chunk = size_t(rows) * n;
    const dim3 grid(rows / kTileRows, n / kTileCols);
    const dim3 block(32, kWarps);

    const int part = n / kParts;
    auto copy_b = [&](int p) {
        const size_t offset = size_t(p) * part * n;
        cudaMemcpyAsync(ctx.b + offset, b.data() + offset, size_t(part) * n * sizeof(float), cudaMemcpyHostToDevice,
                        ctx.copy_stream);
        cudaEventRecord(ctx.b_ready[p], ctx.copy_stream);
    };
    auto launch = [&](int p, int r) {
        cudaStream_t stream = ctx.streams[r];
        cudaStreamWaitEvent(stream, ctx.b_ready[p], 0);
        if (p == 0) {
            GemmKernel<false><<<grid, block, 0, stream>>>(ctx.a + r * chunk, ctx.b, ctx.c + r * chunk, n, 0, part);
        } else {
            GemmKernel<true><<<grid, block, 0, stream>>>(ctx.a + r * chunk, ctx.b, ctx.c + r * chunk, n, p * part,
                                                         (p + 1) * part);
        }
    };

    copy_b(0);
    for (int r = 0; r < chunks; ++r) {
        cudaMemcpyAsync(ctx.a + r * chunk, a.data() + r * chunk, chunk * sizeof(float), cudaMemcpyHostToDevice,
                        ctx.streams[r]);
        launch(0, r);
    }
    for (int p = 1; p < kParts; ++p) {
        copy_b(p);
        for (int r = 0; r < chunks; ++r) {
            launch(p, r);
        }
    }

    builder.join();
    for (int r = 0; r < chunks; ++r) {
        cudaMemcpyAsync(c.data() + r * chunk, ctx.c + r * chunk, chunk * sizeof(float), cudaMemcpyDeviceToHost,
                        ctx.streams[r]);
    }
    for (int r = 0; r < chunks; ++r) {
        cudaStreamSynchronize(ctx.streams[r]);
    }
    return c;
}
