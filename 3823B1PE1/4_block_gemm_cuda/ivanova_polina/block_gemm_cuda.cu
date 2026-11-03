#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>

const int BM = 128;
const int BN = 128;
const int BK = 8;
const int TM = 8;
const int TN = 8;

__global__ void block_gemm_tiled_shared_kernel(const float* __restrict__ a,
                                                const float* __restrict__ b,
                                                float* __restrict__ c,
                                                int n) {

    __shared__ float s_a[BM][BK + 1];
    __shared__ float s_b[BK][BN + 1];

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int thread_id = ty * blockDim.x + tx; 

    int row_base = by * BM + ty * TM;
    int col_base = bx * BN + tx * TN;

    float accum[TM][TN] = {0.0f};

    int num_tiles = n / BK;

    for (int t = 0; t < num_tiles; ++t) {

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int idx = thread_id + i * 256;
            int r = idx / BK;
            int c_idx = idx % BK;
            int global_r = by * BM + r;
            int global_c = t * BK + c_idx;
            s_a[r][c_idx] = (global_r < n && global_c < n) ? a[global_r * n + global_c] : 0.0f;
        }

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int idx = thread_id + i * 256;
            int r = idx / BN;
            int c_idx = idx % BN;
            int global_r = t * BK + r;
            int global_c = bx * BN + c_idx;
            s_b[r][c_idx] = (global_r < n && global_c < n) ? b[global_r * n + global_c] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float reg_a[TM];
            float reg_b[TN];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                reg_a[i] = s_a[ty * TM + i][k];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                reg_b[j] = s_b[k][tx * TN + j];
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    accum[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int r = row_base + i;
            int c_idx = col_base + j;
            if (r < n && c_idx < n) {
                c[r * n + c_idx] = accum[i][j];
            }
        }
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    int size = n * n;
    size_t bytes = size * sizeof(float);
    std::vector<float> c(size, 0.0f);
    if (n == 0) return c;

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(d_a, a.data(), bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, b.data(), bytes, cudaMemcpyHostToDevice, stream);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(n / BN, n / BM);

    block_gemm_tiled_shared_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(d_a, d_b, d_c, n);

    cudaMemcpyAsync(c.data(), d_c, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}