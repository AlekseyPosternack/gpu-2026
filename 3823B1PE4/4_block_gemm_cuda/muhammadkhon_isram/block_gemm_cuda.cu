#include "block_gemm_cuda.h"

#include <cuda_runtime.h>
#include <cstddef>

namespace {

constexpr int kTile = 32;
constexpr int kBlockRows = 8;
constexpr int kPerThread = kTile / kBlockRows;

__global__ void BlockGemmKernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C, int n) {
    __shared__ float As[kTile][kTile];
    __shared__ float Bs[kTile][kTile];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row0 = blockIdx.y * kTile + ty;
    const int col = blockIdx.x * kTile + tx;

    float acc[kPerThread] = {0.f, 0.f, 0.f, 0.f};

    const int tiles = (n + kTile - 1) / kTile;
    for (int t = 0; t < tiles; ++t) {
        #pragma unroll
        for (int r = 0; r < kPerThread; ++r) {
            const int sy = ty + r * kBlockRows;

            const int a_row = blockIdx.y * kTile + sy;
            const int a_col = t * kTile + tx;
            As[sy][tx] = (a_row < n && a_col < n)
                             ? A[static_cast<size_t>(a_row) * n + a_col] : 0.f;

            const int b_row = t * kTile + sy;
            Bs[sy][tx] = (b_row < n && col < n)
                             ? B[static_cast<size_t>(b_row) * n + col] : 0.f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < kTile; ++k) {
            const float bv = Bs[k][tx];
            #pragma unroll
            for (int r = 0; r < kPerThread; ++r)
                acc[r] = fmaf(As[ty + r * kBlockRows][k], bv, acc[r]);
        }
        __syncthreads();
    }

    if (col < n) {
        #pragma unroll
        for (int r = 0; r < kPerThread; ++r) {
            const int row = row0 + r * kBlockRows;
            if (row < n) C[static_cast<size_t>(row) * n + col] = acc[r];
        }
    }
}

struct Context {
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    size_t capacity = 0;
    cudaStream_t stream;

    Context() { cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking); }
    ~Context() {
        Free();
        cudaStreamDestroy(stream);
    }
    void Free() {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
        d_a = d_b = d_c = nullptr;
        capacity = 0;
    }
    void Reserve(size_t elems) {
        if (elems <= capacity) return;
        Free();
        cudaMalloc(&d_a, elems * sizeof(float));
        cudaMalloc(&d_b, elems * sizeof(float));
        cudaMalloc(&d_c, elems * sizeof(float));
        capacity = elems;
    }
};

}  // namespace

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) return {};

    const size_t elems = static_cast<size_t>(n) * n;

    static Context ctx;
    ctx.Reserve(elems);

    cudaMemcpyAsync(ctx.d_a, a.data(), elems * sizeof(float),
                    cudaMemcpyHostToDevice, ctx.stream);
    cudaMemcpyAsync(ctx.d_b, b.data(), elems * sizeof(float),
                    cudaMemcpyHostToDevice, ctx.stream);

    dim3 block(kTile, kBlockRows);
    dim3 grid((n + kTile - 1) / kTile, (n + kTile - 1) / kTile);
    BlockGemmKernel<<<grid, block, 0, ctx.stream>>>(ctx.d_a, ctx.d_b, ctx.d_c, n);

    std::vector<float> c(elems);

    cudaMemcpyAsync(c.data(), ctx.d_c, elems * sizeof(float),
                    cudaMemcpyDeviceToHost, ctx.stream);
    cudaStreamSynchronize(ctx.stream);

    return c;
}