#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BS 32
#define T4 4

__global__ void block_gemm_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int n) {
    __shared__ float As[BS][BS];
    __shared__ float Bs[BS][BS];

    const int tx = threadIdx.x;   
    const int ty = threadIdx.y;   

    const int cRowBase = blockIdx.y * BS + ty * T4;
    const int cCol     = blockIdx.x * BS + tx;

    float acc[T4] = {0.0f};

    const int numK = n / BS;

    for (int kb = 0; kb < numK; ++kb) {
        #pragma unroll
        for (int t = 0; t < T4; ++t) {
            int r = ty * T4 + t;
            As[r][tx] = A[(blockIdx.y * BS + r) * n + (kb * BS + tx)];
        }

        #pragma unroll
        for (int t = 0; t < T4; ++t) {
            int r = ty + t * (BS / T4);
            Bs[r][tx] = B[(kb * BS + r) * n + (blockIdx.x * BS + tx)];
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BS; ++k) {
            float bval = Bs[k][tx];
            #pragma unroll
            for (int t = 0; t < T4; ++t) {
                acc[t] += As[ty * T4 + t][k] * bval;
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int t = 0; t < T4; ++t) {
        C[(cRowBase + t) * n + cCol] = acc[t];
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    std::vector<float> c(n * n, 0.0f);

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    size_t bytes = (size_t)n * n * sizeof(float);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(BS, BS / T4);
    dim3 grid(n / BS, n / BS);

    block_gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}