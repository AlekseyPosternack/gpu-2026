#include "softmax_cuda.h"

#include <cuda_runtime.h>

#define THREADS 256

__global__ void softmax_row_kernel(const float* __restrict__ input,
    float* __restrict__ output,
    int row_length) {
    const int row_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int row_offset = row_idx * row_length;

    __shared__ float reduce_buffer[THREADS];


    float local_max = -INFINITY;
    for (int i = tid; i < row_length; i += blockDim.x) {
        local_max = fmaxf(local_max, input[row_offset + i]);
    }

    reduce_buffer[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_buffer[tid] = fmaxf(reduce_buffer[tid], reduce_buffer[tid + stride]);
        }
        __syncthreads();
    }
    const float row_max = reduce_buffer[0];

    float local_sum = 0.0f;
    for (int i = tid; i < row_length; i += blockDim.x) {
        local_sum += expf(input[row_offset + i] - row_max);
    }

    reduce_buffer[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_buffer[tid] += reduce_buffer[tid + stride];
        }
        __syncthreads();
    }
    const float row_sum = reduce_buffer[0];
    const float inv_sum = 1.0f / row_sum;

    for (int i = tid; i < row_length; i += blockDim.x) {
        output[row_offset + i] = expf(input[row_offset + i] - row_max) * inv_sum;
    }
}

namespace {
    float* g_dev_input = nullptr;
    float* g_dev_output = nullptr;
    std::size_t g_dev_capacity = 0;

    bool ensure_device_memory(std::size_t elements) {
        if (elements <= g_dev_capacity) return true;

        if (g_dev_input) cudaFree(g_dev_input);
        if (g_dev_output) cudaFree(g_dev_output);

        std::size_t bytes = elements * sizeof(float);
        if (cudaMalloc(&g_dev_input, bytes) != cudaSuccess) return false;
        if (cudaMalloc(&g_dev_output, bytes) != cudaSuccess) {
            cudaFree(g_dev_input);
            return false;
        }
        g_dev_capacity = elements;
        return true;
    }
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (input.empty() || row_count <= 0) return {};

    const int row_length = static_cast<int>(input.size() / row_count);
    const std::size_t total_elements = input.size();
    const std::size_t bytes = total_elements * sizeof(float);

    if (!ensure_device_memory(total_elements)) return {};

    std::vector<float> output(total_elements);

    if (cudaMemcpy(g_dev_input, input.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        return {};
    }

    softmax_row_kernel << <row_count, THREADS >> > (g_dev_input, g_dev_output, row_length);

    if (cudaGetLastError() != cudaSuccess) return {};
    if (cudaDeviceSynchronize() != cudaSuccess) return {};

    if (cudaMemcpy(output.data(), g_dev_output, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        return {};
    }

    return output;
}