#include "block_gemm_cuda.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <thread>
#include <vector>

namespace {

constexpr int kBM = 64;
constexpr int kBN = 64;
constexpr int kBK = 16;
constexpr int kTM = 4;
constexpr int kTN = 4;
constexpr int kThreadsX = kBN / kTN;
constexpr int kThreadsY = kBM / kTM;

__global__ void __launch_bounds__(256)
BlockGemmFastKernel(const float* __restrict__ a,
                    const float* __restrict__ b,
                    float* __restrict__ c,
                    int n) {
    __shared__ __align__(16) float as[kBM][kBK];
    __shared__ __align__(16) float bs[kBK][kBN];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int a_row = tid / (kBK / 4);
    const int a_col = (tid % (kBK / 4)) * 4;
    const int b_row = tid / (kBN / 4);
    const int b_col = (tid % (kBN / 4)) * 4;

    const float* a_ptr = a + (blockIdx.y * kBM + a_row) * n + a_col;
    const float* b_ptr = b + b_row * n + blockIdx.x * kBN + b_col;

    float acc[kTM][kTN] = {};

    for (int t = 0; t < n; t += kBK) {
        const float4 va = *reinterpret_cast<const float4*>(a_ptr + t);
        as[a_row][a_col + 0] = va.x;
        as[a_row][a_col + 1] = va.y;
        as[a_row][a_col + 2] = va.z;
        as[a_row][a_col + 3] = va.w;

        const float4 vb = *reinterpret_cast<const float4*>(b_ptr + t * n);
        bs[b_row][b_col + 0] = vb.x;
        bs[b_row][b_col + 1] = vb.y;
        bs[b_row][b_col + 2] = vb.z;
        bs[b_row][b_col + 3] = vb.w;

        __syncthreads();

#pragma unroll
        for (int k = 0; k < kBK; ++k) {
            float av[kTM];
            float bv[kTN];
#pragma unroll
            for (int i = 0; i < kTM; ++i) {
                av[i] = as[ty * kTM + i][k];
            }
#pragma unroll
            for (int j = 0; j < kTN; ++j) {
                bv[j] = bs[k][tx * kTN + j];
            }
#pragma unroll
            for (int i = 0; i < kTM; ++i) {
#pragma unroll
                for (int j = 0; j < kTN; ++j) {
                    acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

    const int row0 = blockIdx.y * kBM + ty * kTM;
    const int col0 = blockIdx.x * kBN + tx * kTN;
#pragma unroll
    for (int i = 0; i < kTM; ++i) {
        *reinterpret_cast<float4*>(c + (row0 + i) * n + col0) =
            make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
    }
}

constexpr int kSmallTile = 16;

__global__ void BlockGemmSmallKernel(const float* a, const float* b, float* c, int n) {
    __shared__ float as[kSmallTile][kSmallTile];
    __shared__ float bs[kSmallTile][kSmallTile];

    const int row = blockIdx.y * kSmallTile + threadIdx.y;
    const int col = blockIdx.x * kSmallTile + threadIdx.x;

    float sum = 0.0f;
    for (int t = 0; t < n; t += kSmallTile) {
        const int ka = t + threadIdx.x;
        const int kb = t + threadIdx.y;
        as[threadIdx.y][threadIdx.x] = (row < n && ka < n) ? a[row * n + ka] : 0.0f;
        bs[threadIdx.y][threadIdx.x] = (kb < n && col < n) ? b[kb * n + col] : 0.0f;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < kSmallTile; ++k) {
            sum = fmaf(as[threadIdx.y][k], bs[k][threadIdx.x], sum);
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

struct DeviceMemory {
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
    std::size_t capacity = 0;

    void Ensure(std::size_t count) {
        if (count > capacity) {
            if (a != nullptr) cudaFree(a);
            if (b != nullptr) cudaFree(b);
            if (c != nullptr) cudaFree(c);

            cudaMalloc(&a, count * sizeof(float));
            cudaMalloc(&b, count * sizeof(float));
            cudaMalloc(&c, count * sizeof(float));
            capacity = count;
        }
    }

    ~DeviceMemory() {
        if (a != nullptr) cudaFree(a);
        if (b != nullptr) cudaFree(b);
        if (c != nullptr) cudaFree(c);
    }
};

DeviceMemory g_dev_mem;

}  // namespace

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) {
        return {};
    }

    const std::size_t count = static_cast<std::size_t>(n) * n;
    const std::size_t bytes = count * sizeof(float);

    g_dev_mem.Ensure(count);

    std::vector<float> c;
    std::thread host_alloc([&c, count] { c.resize(count); });

    cudaMemcpy(g_dev_mem.a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(g_dev_mem.b, b.data(), bytes, cudaMemcpyHostToDevice);

    if (n >= kBM && (n % kBM == 0) && (n % kBN == 0)) {
        dim3 block(kThreadsX, kThreadsY);
        dim3 grid(n / kBN, n / kBM);
        BlockGemmFastKernel<<<grid, block>>>(g_dev_mem.a, g_dev_mem.b, g_dev_mem.c, n);
    } else {
        const int blocks = (n + kSmallTile - 1) / kSmallTile;
        BlockGemmSmallKernel<<<dim3(blocks, blocks), dim3(kSmallTile, kSmallTile)>>>(
            g_dev_mem.a, g_dev_mem.b, g_dev_mem.c, n);
    }

    host_alloc.join();

    cudaMemcpy(c.data(), g_dev_mem.c, bytes, cudaMemcpyDeviceToHost);

    return c;
}
