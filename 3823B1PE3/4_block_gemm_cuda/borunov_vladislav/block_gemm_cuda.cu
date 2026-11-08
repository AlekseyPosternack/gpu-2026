#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <cstring>
#include <thread>

namespace {

constexpr int kBM = 64;
constexpr int kBN = 64;
constexpr int kBK = 16;
constexpr int kTM = 4;
constexpr int kTN = 4;


constexpr int kBDX = kBN / kTN;
constexpr int kBDY = kBM / kTM;

__global__ void __launch_bounds__(256)
BlockGemmKernel(const float* __restrict__ A,
                const float* __restrict__ B,
                float* __restrict__ C, int n) {
    __shared__ __align__(16) float sA[kBM][kBK];
    __shared__ __align__(16) float sB[kBK][kBN];

    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int tid = ty * kBDX + tx;

    const int row0 = blockIdx.y * kBM + ty * kTM;
    const int col0 = blockIdx.x * kBN + tx * kTN;

    const int a_row = tid / (kBK / 4);
    const int a_k   = (tid % (kBK / 4)) * 4;
    const int b_k   = tid / (kBN / 4);
    const int b_col = (tid % (kBN / 4)) * 4;

    const float* a_load = A + (blockIdx.y * kBM + a_row) * n + a_k;
    const float* b_load = B + b_k * n + blockIdx.x * kBN + b_col;

    float acc[kTM][kTN] = {};

    for (int k0 = 0; k0 < n; k0 += kBK) {
        {
            float4 va = __ldg(reinterpret_cast<const float4*>(a_load + k0));
            sA[a_row][a_k + 0] = va.x;
            sA[a_row][a_k + 1] = va.y;
            sA[a_row][a_k + 2] = va.z;
            sA[a_row][a_k + 3] = va.w;
        }
        {
            float4 vb = __ldg(reinterpret_cast<const float4*>(b_load + k0 * n));
            sB[b_k][b_col + 0] = vb.x;
            sB[b_k][b_col + 1] = vb.y;
            sB[b_k][b_col + 2] = vb.z;
            sB[b_k][b_col + 3] = vb.w;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < kBK; ++k) {
            float a_reg[kTM];
#pragma unroll
            for (int i = 0; i < kTM; ++i)
                a_reg[i] = sA[ty * kTM + i][k];

            float b_reg[kTN];
#pragma unroll
            for (int j = 0; j < kTN; ++j)
                b_reg[j] = sB[k][tx * kTN + j];

#pragma unroll
            for (int i = 0; i < kTM; ++i)
#pragma unroll
                for (int j = 0; j < kTN; ++j)
                    acc[i][j] = fmaf(a_reg[i], b_reg[j], acc[i][j]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < kTM; ++i) {
        float4 out = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        *reinterpret_cast<float4*>(C + (row0 + i) * n + col0) = out;
    }
}

constexpr int kFT = 32;

__global__ void BlockGemmFallback(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C, int n) {
    __shared__ float sA[kFT][kFT + 1];  
    __shared__ float sB[kFT][kFT + 1];

    const int tx  = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * kFT + ty;
    const int col = blockIdx.x * kFT + tx;
    float sum = 0.0f;

    for (int t = 0; t < n; t += kFT) {
        sA[ty][tx] = (row < n && t + tx < n) ? __ldg(A + row * n + t + tx) : 0.0f;
        sB[ty][tx] = (t + ty < n && col < n) ? __ldg(B + (t + ty) * n + col) : 0.0f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < kFT; ++k) sum = fmaf(sA[ty][k], sB[k][tx], sum);
        __syncthreads();
    }
    if (row < n && col < n) C[row * n + col] = sum;
}

struct Context {
    float*       d_a    = nullptr;
    float*       d_b    = nullptr;
    float*       d_c    = nullptr;
    float*       h_c    = nullptr;
    size_t       cap    = 0;
    size_t       h_cap  = 0;
    cudaStream_t stream = nullptr;
    const float* reg_a  = nullptr; 
    const float* reg_b  = nullptr;

    void Init() {
        if (stream) return;
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    }

    void Reserve(size_t count, size_t bytes) {
        if (count > cap) {
            cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
            cudaMalloc(&d_a, bytes);
            cudaMalloc(&d_b, bytes);
            cudaMalloc(&d_c, bytes);
            cap = count;
        }
        if (count > h_cap) {
            cudaFreeHost(h_c);
            cudaMallocHost(&h_c, bytes);
            h_cap = count;
        }
    }

    void TryRegister(const float*& slot, const float* ptr, size_t bytes) {
        if (ptr == slot) return;
        if (slot) { cudaHostUnregister(const_cast<float*>(slot)); slot = nullptr; }
        if (cudaHostRegister(const_cast<float*>(ptr), bytes,
                             cudaHostRegisterDefault) == cudaSuccess)
            slot = ptr;
    }
};

Context g_ctx;

}  // namespace


std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                  const std::vector<float>& b,
                                  int n) {
    if (n <= 0) return {};

    const size_t count = size_t(n) * n;
    const size_t bytes = count * sizeof(float);

    g_ctx.Init();
    g_ctx.Reserve(count, bytes);
    g_ctx.TryRegister(g_ctx.reg_a, a.data(), bytes);
    g_ctx.TryRegister(g_ctx.reg_b, b.data(), bytes);

    std::vector<float> output;
    std::thread alloc([&output, count] { output.resize(count); });

    cudaMemcpyAsync(g_ctx.d_a, a.data(), bytes, cudaMemcpyHostToDevice, g_ctx.stream);
    cudaMemcpyAsync(g_ctx.d_b, b.data(), bytes, cudaMemcpyHostToDevice, g_ctx.stream);

    if (n % kBM == 0) {
        dim3 block(kBDX, kBDY);
        dim3 grid(n / kBN, n / kBM);
        BlockGemmKernel<<<grid, block, 0, g_ctx.stream>>>(
            g_ctx.d_a, g_ctx.d_b, g_ctx.d_c, n);
    } else {
        const int tiles = (n + kFT - 1) / kFT;
        BlockGemmFallback<<<dim3(tiles, tiles), dim3(kFT, kFT), 0, g_ctx.stream>>>(
            g_ctx.d_a, g_ctx.d_b, g_ctx.d_c, n);
    }

    cudaMemcpyAsync(g_ctx.h_c, g_ctx.d_c, bytes, cudaMemcpyDeviceToHost, g_ctx.stream);
    cudaStreamSynchronize(g_ctx.stream);

    alloc.join();
    std::memcpy(output.data(), g_ctx.h_c, bytes);
    return output;
}
