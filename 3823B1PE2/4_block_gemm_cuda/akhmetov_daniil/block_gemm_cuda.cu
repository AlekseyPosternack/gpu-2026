#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstddef>

#define BLOCK_SIZE 32
#define TILE_Y 8

__global__ void block_gemm_kernel(const float* a, const float* b, float* c, int n) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int col = bx * BLOCK_SIZE + tx;
    
    int row0 = by * BLOCK_SIZE + ty;
    int row1 = row0 + 8;
    int row2 = row0 + 16;
    int row3 = row0 + 24;
    
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];
    
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    
    for (int k = 0; k < n; k += BLOCK_SIZE) {
        sA[ty][tx]      = a[row0 * n + k + tx];
        sA[ty + 8][tx]  = a[row1 * n + k + tx];
        sA[ty + 16][tx] = a[row2 * n + k + tx];
        sA[ty + 24][tx] = a[row3 * n + k + tx];
        
        sB[ty][tx]      = b[(k + ty) * n + col];
        sB[ty + 8][tx]  = b[(k + ty + 8) * n + col];
        sB[ty + 16][tx] = b[(k + ty + 16) * n + col];
        sB[ty + 24][tx] = b[(k + ty + 24) * n + col];
        
        __syncthreads();
        
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            float b_val = sB[i][tx];
            sum0 += sA[ty][i]      * b_val;
            sum1 += sA[ty + 8][i]  * b_val;
            sum2 += sA[ty + 16][i] * b_val;
            sum3 += sA[ty + 24][i] * b_val;
        }
        
        __syncthreads();
    }
    
    c[row0 * n + col] = sum0;
    c[row1 * n + col] = sum1;
    c[row2 * n + col] = sum2;
    c[row3 * n + col] = sum3;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    size_t bytes = (size_t)n * n * sizeof(float);

    static float* d_a = nullptr;
    static float* d_b = nullptr;
    static float* d_c = nullptr;
    static size_t cached_bytes = 0;
    static cudaStream_t stream;
    static bool initialized = false;

    if (!initialized) {
        cudaStreamCreate(&stream);
        initialized = true;
    }

    if (cached_bytes < bytes) {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);
        cached_bytes = bytes;
    }

    cudaMemcpyAsync(d_a, a.data(), bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, b.data(), bytes, cudaMemcpyHostToDevice, stream);

    dim3 block(32, 8);
    dim3 grid(n / 32, n / 32);

    block_gemm_kernel<<<grid, block, 0, stream>>>(d_a, d_b, d_c, n);

    std::vector<float> c_out((size_t)n * n);

    cudaMemcpyAsync(c_out.data(), d_c, bytes, cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);

    return c_out;
}