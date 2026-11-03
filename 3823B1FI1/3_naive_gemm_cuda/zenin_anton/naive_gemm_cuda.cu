#include "naive_gemm_cuda.h"

#include <cstdio>
#include <cstdlib>
#include <thread>

#include <cuda_runtime.h>


#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                        \
            std::fprintf(stderr, "CUDA error \"%s\" at %s:%d\n",          \
                         cudaGetErrorString(err_), __FILE__, __LINE__);   \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

namespace {

    
    constexpr int kBlockX = 32;          
    constexpr int kBlockY = 8;           
    constexpr int kRowsPerThread = 4;    
    constexpr int kTile = kBlockY * kRowsPerThread;  

    
    __global__ void NaiveGemmKernel(const float* __restrict__ a,
        const float* __restrict__ b,
        float* __restrict__ c,
        int n) {
        const int col = blockIdx.x * kBlockX + threadIdx.x;
        const int row0 = blockIdx.y * kTile + threadIdx.y;
        if (col >= n) {
            return;
        }

        
        const float* a_rows[kRowsPerThread];
#pragma unroll
        for (int r = 0; r < kRowsPerThread; ++r) {
            a_rows[r] = a + min(row0 + r * kBlockY, n - 1) * n;
        }

        float sum[kRowsPerThread] = {};

#pragma unroll 4
        for (int k = 0; k < n; ++k) {
            
            const float b_val = b[k * n + col];
#pragma unroll
            for (int r = 0; r < kRowsPerThread; ++r) {
                sum[r] += a_rows[r][k] * b_val;
            }
        }

#pragma unroll
        for (int r = 0; r < kRowsPerThread; ++r) {
            const int row = row0 + r * kBlockY;
            if (row < n) {
                c[row * n + col] = sum[r];  
            }
        }
    }

    
    struct DeviceBuffer {
        float* ptr = nullptr;
        size_t capacity = 0;

        float* Get(size_t count) {
            if (count > capacity) {
                if (ptr != nullptr) {
                    CUDA_CHECK(cudaFree(ptr));
                }
                CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(float)));
                capacity = count;
            }
            return ptr;
        }

        ~DeviceBuffer() {
            if (ptr != nullptr) {
                cudaFree(ptr);  
            }
        }
    };

    DeviceBuffer g_a;
    DeviceBuffer g_b;
    DeviceBuffer g_c;

}  // namespace

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    if (n <= 0) {
        return {};
    }
    const size_t count = static_cast<size_t>(n) * n;
    const size_t bytes = count * sizeof(float);

    float* d_a = g_a.Get(count);
    float* d_b = g_b.Get(count);
    float* d_c = g_c.Get(count);

    
    std::vector<float> c;
    std::thread alloc_thread([&c, count] { c.resize(count); });

    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 block(kBlockX, kBlockY);
    const dim3 grid((n + kBlockX - 1) / kBlockX, (n + kTile - 1) / kTile);
    NaiveGemmKernel << <grid, block >> > (d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());

    alloc_thread.join();

    
    CUDA_CHECK(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    return c;
}