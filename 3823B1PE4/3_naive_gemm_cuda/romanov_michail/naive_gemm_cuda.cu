#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <thread>

namespace {

constexpr int BLOCK_SIZE = 16;

__global__ void NaiveGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        #pragma unroll 4
        for (int k = 0; k < n; ++k) {
            sum = fmaf(a[row * n + k], b[k * n + col], sum);
        }
        c[row * n + col] = sum;
    }
}

float* GetDeviceBufferA(std::size_t bytes) {
    static float* ptr = nullptr;
    static std::size_t capacity = 0;
    if (bytes > capacity) {
        if (ptr) cudaFree(ptr);
        cudaMalloc(&ptr, bytes);
        capacity = bytes;
    }
    return ptr;
}

float* GetDeviceBufferB(std::size_t bytes) {
    static float* ptr = nullptr;
    static std::size_t capacity = 0;
    if (bytes > capacity) {
        if (ptr) cudaFree(ptr);
        cudaMalloc(&ptr, bytes);
        capacity = bytes;
    }
    return ptr;
}

float* GetDeviceBufferC(std::size_t bytes) {
    static float* ptr = nullptr;
    static std::size_t capacity = 0;
    if (bytes > capacity) {
        if (ptr) cudaFree(ptr);
        cudaMalloc(&ptr, bytes);
        capacity = bytes;
    }
    return ptr;
}

} // namespace

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) {
        return {};
    }

    const std::size_t elements = static_cast<std::size_t>(n) * n;
    const std::size_t bytes = elements * sizeof(float);

    float* d_a = GetDeviceBufferA(bytes);
    float* d_b = GetDeviceBufferB(bytes);
    float* d_c = GetDeviceBufferC(bytes);

    std::vector<float> c;
    std::thread alloc_thread([&c, elements] {
        c.resize(elements);
    });

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    NaiveGemmKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    alloc_thread.join();

    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    return c;
}