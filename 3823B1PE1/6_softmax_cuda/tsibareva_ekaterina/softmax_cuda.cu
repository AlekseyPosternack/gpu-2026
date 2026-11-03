#include "softmax_cuda.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cfloat>

#define CHECK_ERROR(X)                            \
    do {                                          \
        cudaError_t err = (X);                    \
        if (err != cudaSuccess) {                 \
            printf("Error: %s\n",                 \
                   cudaGetErrorString(err));      \
            exit(0);                              \
        }                                         \
    } while (0)

#define BLOCK_SIZE 1024

__global__ void softmax_kernel(const float* __restrict__ x,
                               float* __restrict__ y,
                               int row_size) {
    __shared__ float red[BLOCK_SIZE];

    int tid = threadIdx.x;
    int row = blockIdx.x;
    int b_rows = row * row_size;
    int el_per_thread = row_size / BLOCK_SIZE;

    float vals[16];
    float lmax = -FLT_MAX;
    for (int k = 0; k < el_per_thread; ++k) {
        vals[k] = x[b_rows + tid + k * BLOCK_SIZE];
        if (vals[k] > lmax) lmax = vals[k];
    }

    red[tid] = lmax;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (red[tid + s] > red[tid]) red[tid] = red[tid + s];
        }
        __syncthreads();
    }
    float rowmax = red[0];
    __syncthreads();

    float lsum = 0.0f;
    for (int k = 0; k < el_per_thread; ++k) {
        vals[k] = __expf(vals[k] - rowmax);
        lsum += vals[k];
    }

    red[tid] = lsum;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            red[tid] += red[tid + s];
        }
        __syncthreads();
    }
    float rowsum = red[0];

    for (int k = 0; k < el_per_thread; ++k) {
        y[b_rows + tid + k * BLOCK_SIZE] = vals[k] / rowsum;
    }
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    std::vector<float> output(input.size());
    if (input.empty() || row_count <= 0) return output;

    int row_size = (int)(input.size() / row_count);
    std::size_t bytes = input.size() * sizeof(float);

    float* d_in = nullptr;
    float* d_out = nullptr;
    CHECK_ERROR(cudaMalloc(&d_in, bytes));
    CHECK_ERROR(cudaMalloc(&d_out, bytes));

    CHECK_ERROR(cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice));

    dim3 grid(row_count);
    dim3 block(BLOCK_SIZE);
    softmax_kernel<<<grid, block>>>(d_in, d_out, row_size);
    CHECK_ERROR(cudaGetLastError());
    CHECK_ERROR(cudaDeviceSynchronize());

    CHECK_ERROR(cudaMemcpy(output.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    CHECK_ERROR(cudaFree(d_in));
    CHECK_ERROR(cudaFree(d_out));

    return output;
}