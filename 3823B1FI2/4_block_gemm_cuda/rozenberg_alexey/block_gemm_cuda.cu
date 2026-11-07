#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <algorithm>

#define MAX_SIZE 32

__global__ void BlockGemmKernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int n,
                                int tile_dim) {
    // Pad the inner dimension by +1 to eliminate shared memory bank conflicts
    __shared__ float sA[MAX_SIZE][MAX_SIZE + 1];
    __shared__ float sB[MAX_SIZE][MAX_SIZE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * tile_dim + ty;
    int col = blockIdx.x * tile_dim + tx;

    // Register accumulator for fast partial product summation
    float sum = 0.0f;

    int num_tiles = n / tile_dim;

    for (int t = 0; t < num_tiles; ++t) {
        // Coalesced load of A and B blocks into shared memory
        sA[ty][tx] = A[row * n + (t * tile_dim + tx)];
        sB[ty][tx] = B[(t * tile_dim + ty) * n + col];

        // Synchronize to ensure the entire tile is loaded
        __syncthreads();

        // Compute BlockA * BlockB and accumulate partial results
        if (tile_dim == MAX_SIZE) {
            #pragma unroll
            for (int k = 0; k < MAX_SIZE; ++k) {
                sum += sA[ty][k] * sB[k][tx];
            }
        } else {
            for (int k = 0; k < tile_dim; ++k) {
                sum += sA[ty][k] * sB[k][tx];
            }
        }

        // Synchronize before loading the next tile to prevent overwriting sA/sB
        __syncthreads();
    }

    // Dump result to global memory
    C[row * n + col] = sum;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) return {};

    size_t bytes = static_cast<size_t>(n) * n * sizeof(float);

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    int tile_dim = std::min(n, MAX_SIZE);
    dim3 threads(tile_dim, tile_dim);
    dim3 blocks(n / tile_dim, n / tile_dim);

    BlockGemmKernel<<<blocks, threads>>>(d_a, d_b, d_c, n, tile_dim);
    
    // Overlap host memory allocation with asynchronous GPU execution.
    std::vector<float> c(n * n);

    // Implicitly waits for kernel completion and copies result back to host
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}