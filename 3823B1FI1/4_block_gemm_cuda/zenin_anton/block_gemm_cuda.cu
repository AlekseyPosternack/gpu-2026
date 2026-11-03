#include "block_gemm_cuda.h"

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

    
    constexpr int kTile = 32;
    constexpr int kRowsPerThread = 4;
    constexpr int kBlockX = kTile;                    
    constexpr int kBlockY = kTile / kRowsPerThread;   
    
    __global__ void BlockGemmKernel(const float* __restrict__ a,
        const float* __restrict__ b,
        float* __restrict__ c,
        int n) {
        __shared__ float a_block[kTile][kTile];
        __shared__ float b_block[kTile][kTile];

        const int tx = threadIdx.x;  
        const int ty = threadIdx.y;  
        const int col = blockIdx.x * kTile + tx;
        const int row_base = blockIdx.y * kTile;

        
        float sum[kRowsPerThread] = {};

        
        for (int k_base = 0; k_base < n; k_base += kTile) {
            
#pragma unroll
            for (int r = 0; r < kRowsPerThread; ++r) {
                const int local_row = ty + r * kBlockY;

                const int a_row = row_base + local_row;
                const int a_col = k_base + tx;
                a_block[local_row][tx] =
                    (a_row < n && a_col < n) ? a[a_row * n + a_col] : 0.0f;

                const int b_row = k_base + local_row;
                b_block[local_row][tx] =
                    (b_row < n && col < n) ? b[b_row * n + col] : 0.0f;
            }

            
            __syncthreads();

            
#pragma unroll
            for (int k = 0; k < kTile; ++k) {
                const float b_val = b_block[k][tx];  
#pragma unroll
                for (int r = 0; r < kRowsPerThread; ++r) {
                    sum[r] += a_block[ty + r * kBlockY][k] * b_val;
                }
            }

            
            __syncthreads();
        }

        
        if (col < n) {
#pragma unroll
            for (int r = 0; r < kRowsPerThread; ++r) {
                const int row = row_base + ty + r * kBlockY;
                if (row < n) {
                    c[row * n + col] = sum[r];
                }
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

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
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
    const dim3 grid((n + kTile - 1) / kTile, (n + kTile - 1) / kTile);
    BlockGemmKernel << <grid, block >> > (d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());

    alloc_thread.join();

    
    CUDA_CHECK(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    return c;
}