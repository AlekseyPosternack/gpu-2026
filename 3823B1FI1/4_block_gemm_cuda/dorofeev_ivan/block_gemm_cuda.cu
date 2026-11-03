#include "block_gemm_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <thread>

namespace {

constexpr int kTile = 128;
constexpr int kDepth = 8;
constexpr int kPad = 4;
constexpr int kThreads = 256;
constexpr int kSmallTile = 16;
constexpr int kStrips = 8;

__device__ __forceinline__ float4 Load4(const float* p) {
    return *reinterpret_cast<const float4*>(p);
}

template <bool kAccumulate>
__global__ void __launch_bounds__(kThreads) GemmKernel(const float* __restrict__ a, const float* __restrict__ b,
                                                       float* __restrict__ c, int n, int k_begin, int k_end) {
    __shared__ __align__(16) float as[2][kDepth][kTile + kPad];
    __shared__ __align__(16) float bs[2][kDepth][kTile];

    const int t = threadIdx.x;
    const int tx = t % 16;
    const int ty = t / 16;
    const int load_row = t / 2;
    const int load_col = (t % 2) * 4;
    const float* a_tile = a + size_t(blockIdx.y * kTile + load_row) * n + load_col;
    const float* b_tile = b + size_t(t / 32) * n + blockIdx.x * kTile + (t % 32) * 4;
    float* c_tile = c + size_t(blockIdx.y * kTile) * n + blockIdx.x * kTile;

    float acc[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float* row = c_tile + size_t((i < 4 ? ty * 4 : 64 + ty * 4) + i % 4) * n;
        const float4 lo = kAccumulate ? Load4(row + tx * 4) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const float4 hi = kAccumulate ? Load4(row + 64 + tx * 4) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        acc[i][0] = lo.x;
        acc[i][1] = lo.y;
        acc[i][2] = lo.z;
        acc[i][3] = lo.w;
        acc[i][4] = hi.x;
        acc[i][5] = hi.y;
        acc[i][6] = hi.z;
        acc[i][7] = hi.w;
    }

    float4 next_a = Load4(a_tile + k_begin);
    float4 next_b = Load4(b_tile + size_t(k_begin) * n);
    auto store = [&](int s) {
        as[s][load_col + 0][load_row] = next_a.x;
        as[s][load_col + 1][load_row] = next_a.y;
        as[s][load_col + 2][load_row] = next_a.z;
        as[s][load_col + 3][load_row] = next_a.w;
        *reinterpret_cast<float4*>(&bs[s][t / 32][(t % 32) * 4]) = next_b;
    };
    store(0);
    __syncthreads();

    for (int k0 = k_begin, s = 0; k0 < k_end; k0 += kDepth, s ^= 1) {
        const bool more = k0 + kDepth < k_end;
        if (more) {
            next_a = Load4(a_tile + k0 + kDepth);
            next_b = Load4(b_tile + size_t(k0 + kDepth) * n);
        }
#pragma unroll
        for (int k = 0; k < kDepth; ++k) {
            const float4 a0 = Load4(&as[s][k][ty * 4]);
            const float4 a1 = Load4(&as[s][k][64 + ty * 4]);
            const float4 b0 = Load4(&bs[s][k][tx * 4]);
            const float4 b1 = Load4(&bs[s][k][64 + tx * 4]);
            const float av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float bv[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
            for (int i = 0; i < 8; ++i) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
                }
            }
        }
        if (more) {
            store(s ^ 1);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float* row = c_tile + size_t((i < 4 ? ty * 4 : 64 + ty * 4) + i % 4) * n;
        *reinterpret_cast<float4*>(row + tx * 4) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        *reinterpret_cast<float4*>(row + 64 + tx * 4) = make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
    }
}

__global__ void SmallGemmKernel(const float* a, const float* b, float* c, int n) {
    __shared__ float as[kSmallTile][kSmallTile];
    __shared__ float bs[kSmallTile][kSmallTile];
    const int row = blockIdx.y * kSmallTile + threadIdx.y;
    const int col = blockIdx.x * kSmallTile + threadIdx.x;
    float sum = 0.0f;
    for (int k0 = 0; k0 < n; k0 += kSmallTile) {
        const int ka = k0 + threadIdx.x;
        const int kb = k0 + threadIdx.y;
        as[threadIdx.y][threadIdx.x] = row < n && ka < n ? a[row * n + ka] : 0.0f;
        bs[threadIdx.y][threadIdx.x] = kb < n && col < n ? b[kb * n + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < kSmallTile; ++k) {
            sum = fmaf(as[threadIdx.y][k], bs[k][threadIdx.x], sum);
        }
        __syncthreads();
    }
    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

struct Context {
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
    size_t capacity = 0;
    cudaStream_t copy_stream;
    cudaStream_t streams[kStrips];
    cudaEvent_t a_ready[kStrips];
    cudaEvent_t b_ready[2];

    Context() {
        cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking);
        for (int i = 0; i < kStrips; ++i) {
            cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
            cudaEventCreateWithFlags(&a_ready[i], cudaEventDisableTiming);
        }
        for (cudaEvent_t& e : b_ready) {
            cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
        }
    }

    ~Context() {
        cudaFree(a);
        cudaFree(b);
        cudaFree(c);
        cudaStreamDestroy(copy_stream);
        for (int i = 0; i < kStrips; ++i) {
            cudaStreamDestroy(streams[i]);
            cudaEventDestroy(a_ready[i]);
        }
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

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    const size_t count = size_t(n) * n;
    const size_t bytes = count * sizeof(float);
    Context& ctx = GetContext();
    ctx.Reserve(count);

    if (n < kTile) {
        std::vector<float> c(count);
        if (n > 0) {
            cudaMemcpy(ctx.a, a.data(), bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(ctx.b, b.data(), bytes, cudaMemcpyHostToDevice);
            const int blocks = (n + kSmallTile - 1) / kSmallTile;
            SmallGemmKernel<<<dim3(blocks, blocks), dim3(kSmallTile, kSmallTile)>>>(ctx.a, ctx.b, ctx.c, n);
            cudaMemcpy(c.data(), ctx.c, bytes, cudaMemcpyDeviceToHost);
        }
        return c;
    }

    std::vector<float> c;
    std::thread builder([&] { c.resize(count); });

    const int rows = std::max(kTile, n / kStrips);
    const int strips = n / rows;
    const size_t strip = size_t(rows) * n;
    const int half = n / 2;
    const size_t half_size = size_t(half) * n;
    const dim3 grid(n / kTile, rows / kTile);

    auto copy_b = [&](int p) {
        cudaMemcpyAsync(ctx.b + p * half_size, b.data() + p * half_size, half_size * sizeof(float),
                        cudaMemcpyHostToDevice, ctx.copy_stream);
        cudaEventRecord(ctx.b_ready[p], ctx.copy_stream);
    };
    auto launch = [&](int p, int r) {
        cudaStream_t stream = ctx.streams[r];
        cudaStreamWaitEvent(stream, ctx.b_ready[p], 0);
        if (p == 0) {
            GemmKernel<false><<<grid, kThreads, 0, stream>>>(ctx.a + r * strip, ctx.b, ctx.c + r * strip, n, 0, half);
        } else {
            GemmKernel<true><<<grid, kThreads, 0, stream>>>(ctx.a + r * strip, ctx.b, ctx.c + r * strip, n, half, n);
        }
    };

    const int middle = (strips - 1) / 2;
    copy_b(0);
    for (int r = 0; r < strips; ++r) {
        cudaMemcpyAsync(ctx.a + r * strip, a.data() + r * strip, strip * sizeof(float), cudaMemcpyHostToDevice,
                        ctx.copy_stream);
        cudaEventRecord(ctx.a_ready[r], ctx.copy_stream);
        cudaStreamWaitEvent(ctx.streams[r], ctx.a_ready[r], 0);
        launch(0, r);
        if (r == middle) {
            copy_b(1);
            for (int q = 0; q <= r; ++q) {
                launch(1, q);
            }
        } else if (r > middle) {
            launch(1, r);
        }
    }

    builder.join();
    for (int r = 0; r < strips; ++r) {
        cudaMemcpyAsync(c.data() + r * strip, ctx.c + r * strip, strip * sizeof(float), cudaMemcpyDeviceToHost,
                        ctx.streams[r]);
    }
    for (int r = 0; r < strips; ++r) {
        cudaStreamSynchronize(ctx.streams[r]);
    }
    return c;
}
