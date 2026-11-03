#include "softmax_cuda.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace {
constexpr int kThreads = 256;

__global__ void SoftmaxKernel(const float* input, float* output, int rows, int cols) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float shared[];
    float* max_values = shared;
    float* sums = shared + blockDim.x;
    float local_max = -3.402823466e+38F;
    for (int col = threadIdx.x; col < cols; col += blockDim.x)
        local_max = fmaxf(local_max, input[row * cols + col]);
    max_values[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            max_values[threadIdx.x] = fmaxf(max_values[threadIdx.x], max_values[threadIdx.x + stride]);
        __syncthreads();
    }
    const float row_max = max_values[0];
    float local_sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x)
        local_sum += expf(input[row * cols + col] - row_max);
    sums[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    const float denominator = sums[0];
    for (int col = threadIdx.x; col < cols; col += blockDim.x)
        output[row * cols + col] = expf(input[row * cols + col] - row_max) / denominator;
}
void CheckCuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (row_count <= 0 || input.size() % static_cast<std::size_t>(row_count) != 0)
        throw std::invalid_argument("row_count must be positive and divide input size");
    const int cols = static_cast<int>(input.size() / row_count);
    std::vector<float> output(input.size());
    if (input.empty()) return output;
    const std::size_t bytes = input.size() * sizeof(float);
    float* device_input = nullptr;
    float* device_output = nullptr;
    CheckCuda(cudaMalloc(&device_input, bytes), "cudaMalloc(input)");
    try {
        CheckCuda(cudaMalloc(&device_output, bytes), "cudaMalloc(output)");
        CheckCuda(cudaMemcpy(device_input, input.data(), bytes, cudaMemcpyHostToDevice), "copy input");
        SoftmaxKernel<<<row_count, kThreads, 2 * kThreads * sizeof(float)>>>(device_input, device_output, row_count, cols);
        CheckCuda(cudaGetLastError(), "SoftmaxKernel launch");
        CheckCuda(cudaMemcpy(output.data(), device_output, bytes, cudaMemcpyDeviceToHost), "copy output");
    } catch (...) {
        cudaFree(device_output);
        cudaFree(device_input);
        throw;
    }
    CheckCuda(cudaFree(device_output), "cudaFree(output)");
    CheckCuda(cudaFree(device_input), "cudaFree(input)");
    return output;
}
