#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <future>
#include <stdexcept>
namespace {
constexpr int depth = 32;

void CheckCUDA(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

struct DeviceBuffer {
    float* data = nullptr;
    std::size_t capacity = 0;
    ~DeviceBuffer() { cudaFree(data); }
    void Reserve(std::size_t count) {
        if (count > capacity) {
            float* next = nullptr;
            CheckCUDA(cudaMalloc(&next, 3 * count * sizeof(float)));
            cudaFree(data);
            data = next;
            capacity = count;
        }
    }
};

__device__ void Accumulate(float4& sum, float a, float4 b) {
    sum.x = fmaf(a, b.x, sum.x);
    sum.y = fmaf(a, b.y, sum.y);
    sum.z = fmaf(a, b.z, sum.z);
    sum.w = fmaf(a, b.w, sum.w);
}

__global__ void BlockKernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n) {
    __shared__ __align__(16) float tileA[depth][68];
    __shared__ __align__(16) float tileB[depth][128];
    const int tid = threadIdx.y * 32 + threadIdx.x;
    const int row = blockIdx.y * 64;
    const int col = blockIdx.x * 128;
    float4 sum[8] = {};
    for (int start = 0; start < n; start += depth) {
#pragma unroll
        for (int i = tid; i < 64 * depth / 4; i += 256) {
            const int r = i / (depth / 4);
            const int k = i % (depth / 4) * 4;
            const float4 v = *reinterpret_cast<const float4*>(a + (row + r) * n + start + k);
            tileA[k][r] = v.x;
            tileA[k + 1][r] = v.y;
            tileA[k + 2][r] = v.z;
            tileA[k + 3][r] = v.w;
        }

#pragma unroll
        for (int i = tid; i < depth * 128 / 4; i += 256) {
            const int k = i / 32;
            const int j = i % 32 * 4;
            *reinterpret_cast<float4*>(&tileB[k][j]) =
                *reinterpret_cast<const float4*>(b + (start + k) * n + col + j);
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < depth; ++k) {
            const float4 a0 = *reinterpret_cast<const float4*>(&tileA[k][threadIdx.y * 8]);
            const float4 a1 = *reinterpret_cast<const float4*>(&tileA[k][threadIdx.y * 8 + 4]);
            const float av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float4 bv = *reinterpret_cast<const float4*>(&tileB[k][threadIdx.x * 4]);

#pragma unroll
            for (int r = 0; r < 8; ++r) {
                Accumulate(sum[r], av[r], bv);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < 8; ++r) {
        *reinterpret_cast<float4*>(c + (row + threadIdx.y * 8 + r) * n + col + threadIdx.x * 4) = sum[r];
    }
}

__global__ void SmallBlockKernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n) {
    __shared__ float tileA[16][16];
    __shared__ float tileB[16][16];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int row = blockIdx.y * 16 + y;
    const int col = blockIdx.x * 16 + x;
    float sum = 0.0f;
    for (int start = 0; start < n; start += 16) {
        tileA[y][x] = row < n && start + x < n ? a[row * n + start + x] : 0.0f;
        tileB[y][x] = start + y < n && col < n ? b[(start + y) * n + col] : 0.0f;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < 16; ++k) {
            sum = fmaf(tileA[y][k], tileB[k][x], sum);
        }
        __syncthreads();
    }
    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    if (n <= 0) {
        return {};
    }
    const std::size_t count = static_cast<std::size_t>(n) * n;
    if (a.size() != count || b.size() != count) {
        throw std::invalid_argument("Matrix size does not match n");
    }
    const std::size_t bytes = count * sizeof(float);
    thread_local DeviceBuffer buffer;
    buffer.Reserve(count);
    float* da = buffer.data;
    float* db = da + count;
    float* dc = db + count;
    auto output = std::async(std::launch::async, [count] { return std::vector<float>(count); });
    CheckCUDA(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
    CheckCUDA(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
    if (n % 128 == 0) {
        BlockKernel<<<dim3(n / 128, n / 64), dim3(32, 8)>>>(da, db, dc, n);
    } else {
        const unsigned blocks = static_cast<unsigned>((n + 15) / 16);
        SmallBlockKernel<<<dim3(blocks, blocks), dim3(16, 16)>>>(da, db, dc, n);
    }
    CheckCUDA(cudaGetLastError());
    std::vector<float> result = output.get();
    CheckCUDA(cudaMemcpy(result.data(), dc, bytes, cudaMemcpyDeviceToHost));
    return result;
}
