#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define TILE_DIM 4
#define BLOCK_SIZE 16

__global__ void naive_gemm_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int n) {

    int row = (blockIdx.y * BLOCK_SIZE + threadIdx.y) * TILE_DIM;
    int col = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * TILE_DIM;

    if (row >= n || col >= n) return;

    float acc[4][4] = {0.0f};

    for (int k = 0; k < n; k += 4) {
        float a_val[4][4];
        float b_val[4][4];

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float4 a_vec = *reinterpret_cast<const float4*>(&A[(row + i) * n + k]);
            a_val[i][0] = a_vec.x;
            a_val[i][1] = a_vec.y;
            a_val[i][2] = a_vec.z;
            a_val[i][3] = a_vec.w;
        }

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float4 b_vec = *reinterpret_cast<const float4*>(&B[(k + i) * n + col]);
            b_val[i][0] = b_vec.x;
            b_val[i][1] = b_vec.y;
            b_val[i][2] = b_vec.z;
            b_val[i][3] = b_vec.w;
        }

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                #pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    acc[i][j] += a_val[i][kk] * b_val[kk][j];
                }
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float4 c_vec;
        c_vec.x = acc[i][0];
        c_vec.y = acc[i][1];
        c_vec.z = acc[i][2];
        c_vec.w = acc[i][3];
        *reinterpret_cast<float4*>(&C[(row + i) * n + col]) = c_vec;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    std::vector<float> c(n * n, 0.0f);

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    size_t bytes = n * n * sizeof(float);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n + (BLOCK_SIZE * TILE_DIM) - 1) / (BLOCK_SIZE * TILE_DIM),
              (n + (BLOCK_SIZE * TILE_DIM) - 1) / (BLOCK_SIZE * TILE_DIM));

    naive_gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}