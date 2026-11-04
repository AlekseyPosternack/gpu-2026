#include "softmax_cuda.h"
#include <cuda_runtime.h>
#include <cmath>

__global__ void softmax_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int row_count,
                               int row_size) {
    int row = blockIdx.x;
    if (row >= row_count) return;

    extern __shared__ float shared_data[];

    // 1. Находим максимум в строке (для численной стабильности)
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        float val = input[row * row_size + i];
        if (val > max_val) max_val = val;
    }

    shared_data[threadIdx.x] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (shared_data[threadIdx.x + s] > shared_data[threadIdx.x]) {
                shared_data[threadIdx.x] = shared_data[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    float row_max = shared_data[0];
    __syncthreads();

    float sum = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        sum += expf(input[row * row_size + i] - row_max);
    }

    shared_data[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + s];
        }
        __syncthreads();
    }
    float row_sum = shared_data[0];

    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        output[row * row_size + i] = expf(input[row * row_size + i] - row_max) / row_sum;
    }
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (input.empty() || row_count == 0) return {};
    
    int row_size = input.size() / row_count;
    std::vector<float> output(input.size(), 0.0f);

    const size_t bytes = input.size() * sizeof(float);
    float *d_input = nullptr, *d_output = nullptr;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    if (row_size < 256) threadsPerBlock = row_size; 

    size_t sharedMemSize = threadsPerBlock * sizeof(float);

    softmax_kernel<<<row_count, threadsPerBlock, sharedMemSize>>>(d_input, d_output, row_count, row_size);
    cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}