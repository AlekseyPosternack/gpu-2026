#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>
#include <cstddef>

namespace {

constexpr int kBlockX = 32;
constexpr int kBlockY = 8;

__global__ void GemmVec4Kernel(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C, int n) {
    const int col4 = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int n4 = n >> 2;
    if (row >= n || col4 >= n4) return;

    const float4* A4 = reinterpret_cast<const float4*>(A + static_cast<size_t>(row) * n);
    const float4* B4 = reinterpret_cast<const float4*>(B);

    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);

    #pragma unroll 2
    for (int k4 = 0; k4 < n4; ++k4) {
        const float4 a = A4[k4];
        const size_t k = static_cast<size_t>(k4) * 4;

        const float4 b0 = B4[(k + 0) * n4 + col4];
        const float4 b1 = B4[(k + 1) * n4 + col4];
        const float4 b2 = B4[(k + 2) * n4 + col4];
        const float4 b3 = B4[(k + 3) * n4 + col4];

        acc.x = fmaf(a.x, b0.x, acc.x); acc.y = fmaf(a.x, b0.y, acc.y);
        acc.z = fmaf(a.x, b0.z, acc.z); acc.w = fmaf(a.x, b0.w, acc.w);

        acc.x = fmaf(a.y, b1.x, acc.x); acc.y = fmaf(a.y, b1.y, acc.y);
        acc.z = fmaf(a.y, b1.z, acc.z); acc.w = fmaf(a.y, b1.w, acc.w);

        acc.x = fmaf(a.z, b2.x, acc.x); acc.y = fmaf(a.z, b2.y, acc.y);
        acc.z = fmaf(a.z, b2.z, acc.z); acc.w = fmaf(a.z, b2.w, acc.w);

        acc.x = fmaf(a.w, b3.x, acc.x); acc.y = fmaf(a.w, b3.y, acc.y);
        acc.z = fmaf(a.w, b3.z, acc.z); acc.w = fmaf(a.w, b3.w, acc.w);
    }

    reinterpret_cast<float4*>(C + static_cast<size_t>(row) * n)[col4] = acc;
}

__global__ void GemmScalarKernel(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C, int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n || col >= n) return;

    float acc = 0.f;
    for (int k = 0; k < n; ++k)
        acc = fmaf(A[static_cast<size_t>(row) * n + k],
                   B[static_cast<size_t>(k) * n + col], acc);
    C[static_cast<size_t>(row) * n + col] = acc;
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

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
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

    if (n % 4 == 0) {
        const int n4 = n / 4;
        dim3 block(kBlockX, kBlockY);
        dim3 grid((n4 + kBlockX - 1) / kBlockX, (n + kBlockY - 1) / kBlockY);
        GemmVec4Kernel<<<grid, block, 0, ctx.stream>>>(ctx.d_a, ctx.d_b, ctx.d_c, n);
    } else {
        dim3 block(32, 8);
        dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
        GemmScalarKernel<<<grid, block, 0, ctx.stream>>>(ctx.d_a, ctx.d_b, ctx.d_c, n);
    }

    std::vector<float> c(elems);

    cudaMemcpyAsync(c.data(), ctx.d_c, elems * sizeof(float),
                    cudaMemcpyDeviceToHost, ctx.stream);
    cudaStreamSynchronize(ctx.stream);

    return c;
}