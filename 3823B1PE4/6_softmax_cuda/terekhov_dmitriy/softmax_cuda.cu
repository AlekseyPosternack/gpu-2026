#include "softmax_cuda.h"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstddef>
#include <stdexcept>

namespace {

constexpr int thread_count = 256;

void checkCuda(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

__global__ void softmaxKernel(float* data, std::size_t row_size) {
    __shared__ float values[thread_count];
    const int tid = threadIdx.x;
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * row_size;

    float maximum = -CUDART_INF_F;
    for (std::size_t col = tid; col < row_size; col += blockDim.x) {
        maximum = fmaxf(maximum, data[start + col]);
    }
    values[tid] = maximum;
    __syncthreads();

    for (int step = thread_count / 2; step > 0; step /= 2) {
        if (tid < step) {
            values[tid] = fmaxf(values[tid], values[tid + step]);
        }
        __syncthreads();
    }
    const float row_max = values[0];
    __syncthreads();

    float sum = 0.0f;
    for (std::size_t col = tid; col < row_size; col += blockDim.x) {
        const float value = expf(data[start + col] - row_max);
        data[start + col] = value;
        sum += value;
    }
    values[tid] = sum;
    __syncthreads();

    for (int step = thread_count / 2; step > 0; step /= 2) {
        if (tid < step) {
            values[tid] += values[tid + step];
        }
        __syncthreads();
    }
    const float row_sum = values[0];
    for (std::size_t col = tid; col < row_size; col += blockDim.x) {
        data[start + col] /= row_sum;
    }
}

}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (row_count <= 0 || input.size() % row_count != 0) {
        throw std::invalid_argument("Invalid row count or matrix dimensions");
    }
    if (input.empty()) {
        return {};
    }

    const std::size_t row_size = input.size() / row_count;
    const std::size_t bytes = input.size() * sizeof(float);
    float* device_data = nullptr;
    checkCuda(cudaMalloc(&device_data, bytes));

    std::vector<float> result;
    try {
        checkCuda(cudaMemcpy(device_data, input.data(), bytes, cudaMemcpyHostToDevice));
        softmaxKernel<<<row_count, thread_count>>>(device_data, row_size);
        checkCuda(cudaGetLastError());

        result.resize(input.size());
        checkCuda(cudaMemcpy(result.data(), device_data, bytes, cudaMemcpyDeviceToHost));
    } catch (...) {
        cudaFree(device_data);
        throw;
    }

    checkCuda(cudaFree(device_data));
    return result;
}
