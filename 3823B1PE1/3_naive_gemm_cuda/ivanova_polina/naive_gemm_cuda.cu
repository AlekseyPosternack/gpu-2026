#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define BM 64
#define BN 64
#define BK 8
#define TM 4
#define TN 4

__global__ void __launch_bounds__(256) gemm_optimized_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N) 
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tid = threadIdx.y * blockDim.x + threadIdx.x; // 0..255

    __shared__ float sA[BK][BM];
    __shared__ float sB[BK][BN];

    int threadRow = tid / (BN / TN); // 0..15
    int threadCol = tid % (BN / TN); // 0..15

    int globalRowStart = by * BM;
    int globalColStart = bx * BN;

    float accum[TM][TN] = {0.0f};

    int a_inner_row = tid / 4;        // 0..63 (строка внутри BM)
    int a_inner_col = (tid % 4) * 2;  // 0, 2, 4, 6 (колонка внутри BK)

    int b_inner_row = tid / 32;       // 0..7 (строка внутри BK)
    int b_inner_col = (tid % 32) * 2; // 0..62 по 2 элемента (колонка внутри BN)

    for (int bk = 0; bk < N; bk += BK) {

        for (int i = 0; i < 2; ++i) {
            int r = globalRowStart + a_inner_row;
            int c = bk + a_inner_col + i;
            sA[a_inner_col + i][a_inner_row] = (r < N && c < N) ? A[r * N + c] : 0.0f;
        }

        for (int i = 0; i < 2; ++i) {
            int r = bk + b_inner_row;
            int c = globalColStart + b_inner_col + i;
            sB[b_inner_row][b_inner_col + i] = (r < N && c < N) ? B[r * N + c] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float regA[TM];
            float regB[TN];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                regA[i] = sA[k][threadRow * TM + i];
            }

            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                regB[j] = sB[k][threadCol * TN + j];
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    accum[i][j] = fmaf(regA[i], regB[j], accum[i][j]);
                }
            }
        }

        __syncthreads();
    }

    int outRowBase = globalRowStart + threadRow * TM;
    int outColBase = globalColStart + threadCol * TN;

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int row = outRowBase + i;
            int col = outColBase + j;
            if (row < N && col < N) {
                C[row * N + col] = accum[i][j];
            }
        }
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    size_t size = static_cast<size_t>(n) * n;
    size_t bytes = size * sizeof(float);
    std::vector<float> c(size, 0.0f);

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaHostRegister((void*)a.data(), bytes, cudaHostRegisterReadOnly);
    cudaHostRegister((void*)b.data(), bytes, cudaHostRegisterReadOnly);
    cudaHostRegister((void*)c.data(), bytes, cudaHostRegisterDefault);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(d_a, a.data(), bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, b.data(), bytes, cudaMemcpyHostToDevice, stream);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + BN - 1) / BN, (n + BM - 1) / BM);

    gemm_optimized_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(d_a, d_b, d_c, n);

    cudaMemcpyAsync(c.data(), d_c, bytes, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);
    cudaHostUnregister((void*)a.data());
    cudaHostUnregister((void*)b.data());
    cudaHostUnregister((void*)c.data());

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}