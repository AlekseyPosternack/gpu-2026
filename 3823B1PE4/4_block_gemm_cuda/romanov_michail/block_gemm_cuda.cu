#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <thread>

namespace {

constexpr int BLOCK_SIZE = 32;

__global__ void BlockGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                int n) {
    __shared__ float s_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_b[BLOCK_SIZE][BLOCK_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * BLOCK_SIZE + ty;
    const int col = blockIdx.x * BLOCK_SIZE + tx;

    float sum = 0.0f;

    for (int k_block = 0; k_block < n; k_block += BLOCK_SIZE) {
        if (row < n && (k_block + tx) < n) {
            s_a[ty][tx] = a[row * n + k_block + tx];
        } else {
            s_a[ty][tx] = 0.0f;
        }

        if ((k_block + ty) < n && col < n) {
            s_b[ty][tx] = b[(k_block + ty) * n + col];
        } else {
            s_b[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum = fmaf(s_a[ty][k], s_b[k][tx], sum);
        }

        __syncthreads();
    }

    if (row < n && col < n) {
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

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
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

    BlockGemmKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    alloc_thread.join();

    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    return c;
}