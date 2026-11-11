#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

const int BM = 64;
const int BN = 64;
const int BK = 8;
const int TM = 8;
const int TN = 8;

__global__ void optimized_block_gemm_kernel(const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float* __restrict__ C,
                                           int M, int N, int K) {
    __shared__ float s_a[BM][BK + 1];
    __shared__ float s_b[BK][BN + 1];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int thread_id = ty * blockDim.x + tx;

    int load_a_row = thread_id / BK;
    int load_a_col = thread_id % BK;
    int load_b_row = thread_id / BN;
    int load_b_col = thread_id % BN;

    float thread_results[TM][TN] = {0.0f};
    float reg_a[TM];
    float reg_b[TN];

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        if (load_a_row < BM && (bkIdx + load_a_col) < K) {
            s_a[load_a_row][load_a_col] = A[(by * BM + load_a_row) * K + bkIdx + load_a_col];
        } else {
            s_a[load_a_row][load_a_col] = 0.0f;
        }

        if (load_b_row < BK && (bx * BN + load_b_col) < N) {
            s_b[load_b_row][load_b_col] = B[(bkIdx + load_b_row) * N + bx * BN + load_b_col];
        } else {
            s_b[load_b_row][load_b_col] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
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
                    thread_results[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int row = by * BM + ty * TM + i;
        if (row < M) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int col = bx * BN + tx * TN + j;
                if (col < N) {
                    C[row * N + col] = thread_results[i][j];
                }
            }
        }
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    size_t size_bytes = n * n * sizeof(float);
    std::vector<float> c(n * n, 0.0f);

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;

    cudaMalloc(&d_a, size_bytes);
    cudaMalloc(&d_b, size_bytes);
    cudaMalloc(&d_c, size_bytes);

    cudaMemcpy(d_a, a.data(), size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size_bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BN / TN, BM / TM);
    dim3 blocksPerGrid((n + BN - 1) / BN, (n + BM - 1) / BM);

    optimized_block_gemm_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n, n, n);
    cudaDeviceSynchronize();

    cudaMemcpy(c.data(), d_c, size_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}