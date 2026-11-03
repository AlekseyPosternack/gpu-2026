#include <cuda_runtime.h>
#include <cmath>

#include "softmax_cuda.h"

namespace {
constexpr float kLog2e = 1.44269504089f;
constexpr int kBlockSize = 256;
}

__global__ void softmaxKernel(const float* __restrict__ input, float* __restrict__ output, int cols) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    __shared__ float s_reduce[kBlockSize];

    // Thread-local max + Shared memory reduction
    float local_max = -INFINITY;
    for (int i = tid; i < cols; i += kBlockSize) {
        local_max = fmaxf(local_max, row_in[i]);
    }
    
    s_reduce[tid] = local_max;
    __syncthreads();

    #pragma unroll
    for (int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_reduce[tid] = fmaxf(s_reduce[tid], s_reduce[tid + stride]);
        }
        __syncthreads();
    }
    const float row_max = s_reduce[0];

    // Thread-local sum of exponentials + Shared memory reduction
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += kBlockSize) {
        local_sum += exp2f((row_in[i] - row_max) * kLog2e);
    }

    s_reduce[tid] = local_sum;
    __syncthreads();

    #pragma unroll
    for (int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_reduce[tid] += s_reduce[tid + stride];
        }
        __syncthreads();
    }
    const float row_sum = s_reduce[0];
    const float inv_sum = 1.0f / row_sum;

    // Normalize and write to global memory
    for (int i = tid; i < cols; i += kBlockSize) {
        row_out[i] = exp2f((row_in[i] - row_max) * kLog2e) * inv_sum;
    }
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (input.empty() || row_count == 0) {
        return {};
    }

    const size_t size = input.size();
    const int cols = size / row_count;
    const size_t bytes = size * sizeof(float);

    static float* d_in = nullptr;
    static float* d_out = nullptr;
    static size_t current_capacity = 0;

    if (current_capacity < size) {
        if (d_in) cudaFree(d_in);
        if (d_out) cudaFree(d_out);
        cudaMalloc(&d_in, bytes);
        cudaMalloc(&d_out, bytes);
        current_capacity = size;
    }

    cudaMemcpyAsync(d_in, input.data(), bytes, cudaMemcpyHostToDevice);

    softmaxKernel<<<row_count, kBlockSize>>>(d_in, d_out, cols);

    std::vector<float> result(size);

    cudaMemcpyAsync(result.data(), d_out, bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    return result;
}
