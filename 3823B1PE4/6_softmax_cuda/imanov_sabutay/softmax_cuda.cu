#include "softmax_cuda.h"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void CheckCUDA(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

struct DeviceMemory {
    float* data = nullptr;
    DeviceMemory() = default;
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;
    ~DeviceMemory() {
        if (data != nullptr) cudaFree(data);
    }
};

__global__ void SoftmaxKernel(float* data, std::size_t columns, int rows) {
    __shared__ float reduction[256];
    const unsigned int tid = threadIdx.x;
    for (std::size_t row = blockIdx.x; row < static_cast<std::size_t>(rows); row += gridDim.x) {
        float* values = data + row * columns;
        float maximum = -CUDART_INF_F;
        for (std::size_t col = tid; col < columns; col += blockDim.x) {
            maximum = fmaxf(maximum, values[col]);
        }
        reduction[tid] = maximum;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (tid < stride) reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
            __syncthreads();
        }
        maximum = reduction[0];
        // Finish reading the maximum before reusing the reduction array.
        __syncthreads();

        float sum = 0.0f;
        for (std::size_t col = tid; col < columns; col += blockDim.x) {
            const float value = __expf(values[col] - maximum);
            values[col] = value; // Cache exponentials; do not compute them twice.
            sum += value;
        }
        reduction[tid] = sum;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (tid < stride) reduction[tid] += reduction[tid + stride];
            __syncthreads();
        }
        const float inverse_sum = 1.0f / reduction[0];
        for (std::size_t col = tid; col < columns; col += blockDim.x) {
            values[col] *= inverse_sum;
        }
        // Protect reduction[0] until all threads finish using this row's sum.
        __syncthreads();
    }
}

} // namespace

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (row_count <= 0) throw std::invalid_argument("row_count must be positive");
    const std::size_t rows = static_cast<std::size_t>(row_count);
    if (input.size() % rows != 0) {
        throw std::invalid_argument("input size must be divisible by row_count");
    }
    if (input.empty()) return {};
    if (input.size() > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::length_error("matrix size is too large");
    }
    const std::size_t columns = input.size() / rows;
    const std::size_t bytes = input.size() * sizeof(float);
    DeviceMemory memory;
    CheckCUDA(cudaMalloc(reinterpret_cast<void**>(&memory.data), bytes), "cudaMalloc");
    CheckCUDA(cudaMemcpy(memory.data, input.data(), bytes, cudaMemcpyHostToDevice), "copy input to GPU");

    unsigned int threads = 32;
    while (threads < columns && threads < 256) threads *= 2;
    const unsigned int blocks = static_cast<unsigned int>(std::min<std::size_t>(rows, 65535));
    SoftmaxKernel<<<blocks, threads>>>(memory.data, columns, row_count);
    CheckCUDA(cudaGetLastError(), "SoftmaxKernel launch");

    // Allocate the result while the kernel may still be executing.
    std::vector<float> output(input.size());
    CheckCUDA(cudaMemcpy(output.data(), memory.data, bytes, cudaMemcpyDeviceToHost), "copy output from GPU");
    return output;
}
