#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <future>
#include <stdexcept>

namespace {

constexpr int rowsPerThread = 32;

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

__device__ void Accumulate(float2& sum, float a, float2 b) {
    sum.x = fmaf(a, b.x, sum.x);
    sum.y = fmaf(a, b.y, sum.y);
}

__global__ void GemmKernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n) {
    const int row = blockIdx.y * (4 * rowsPerThread) + threadIdx.y;
    const int col = 2 * (blockIdx.x * 32 + threadIdx.x);
    float2 sum[rowsPerThread] = {};

    for (int k = 0; k < n; k += 4) {
        const float2 b0 = *reinterpret_cast<const float2*>(b + k * n + col);
        const float2 b1 = *reinterpret_cast<const float2*>(b + (k + 1) * n + col);
        const float2 b2 = *reinterpret_cast<const float2*>(b + (k + 2) * n + col);
        const float2 b3 = *reinterpret_cast<const float2*>(b + (k + 3) * n + col);

#pragma unroll
        for (int r = 0; r < rowsPerThread; ++r) {
            const float4 av = *reinterpret_cast<const float4*>(a + (row + 4 * r) * n + k);
            Accumulate(sum[r], av.x, b0);
            Accumulate(sum[r], av.y, b1);
            Accumulate(sum[r], av.z, b2);
            Accumulate(sum[r], av.w, b3);
        }
    }

#pragma unroll
    for (int r = 0; r < rowsPerThread; ++r) {
        *reinterpret_cast<float2*>(c + (row + 4 * r) * n + col) = sum[r];
    }
}

__global__ void SmallGemmKernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n) {
    const int row = blockIdx.y * 16 + threadIdx.y;
    const int col = blockIdx.x * 16 + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum = fmaf(a[row * n + k], b[k * n + col], sum);
        }
        c[row * n + col] = sum;
    }
}
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
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
    if (n % 64 == 0 && n % (4 * rowsPerThread) == 0) {
        GemmKernel<<<dim3(n / 64, n / (4 * rowsPerThread)), dim3(32, 4)>>>(da, db, dc, n);
    } else {
        const unsigned blocks = static_cast<unsigned>((n + 15) / 16);
        SmallGemmKernel<<<dim3(blocks, blocks), dim3(16, 16)>>>(da, db, dc, n);
    }
    CheckCUDA(cudaGetLastError());
    std::vector<float> result = output.get();
    CheckCUDA(cudaMemcpy(result.data(), dc, bytes, cudaMemcpyDeviceToHost));
    return result;
}
