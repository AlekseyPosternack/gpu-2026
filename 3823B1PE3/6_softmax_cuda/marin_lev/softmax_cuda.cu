#include "softmax_cuda.h"

#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace {

void Check(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

struct GpuBuffer {
    float* data = nullptr;
    std::size_t capacity = 0;

    ~GpuBuffer() {
        cudaFree(data);
    }
};

__device__ __forceinline__ float ReduceWarp(float value, bool find_max) {
#pragma unroll
    for (int step = 16; step > 0; step /= 2) {
        float other = __shfl_down_sync(0xffffffffu, value, step);

        if (find_max) {
            value = fmaxf(value, other);
        } else {
            value += other;
        }
    }
    return value;
}

__device__ __forceinline__ float ReduceBlock(float value, float* partial,
                                            bool find_max) {
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    value = ReduceWarp(value, find_max);

    if (lane == 0) {
        partial[warp] = value;
    }
    __syncthreads();

    value = find_max ? -CUDART_INF_F : 0.0f;

    if (threadIdx.x < blockDim.x / 32) {
        value = partial[lane];
    }

    if (warp == 0) {
        value = ReduceWarp(value, find_max);

        if (lane == 0) {
            partial[0] = value;
        }
    }
    __syncthreads();

    float result = partial[0];
    __syncthreads();

    return result;
}

__global__ void SoftmaxKernel(const float* __restrict__ input,
                              float* __restrict__ output,
                              std::size_t columns) {
    extern __shared__ float values[];
    __shared__ float partial[8];

    std::size_t start = static_cast<std::size_t>(blockIdx.x) * columns;
    bool use_shared = columns <= 8192;

    float maximum = -CUDART_INF_F;

    for (std::size_t col = threadIdx.x; col < columns; col += blockDim.x) {
        float x = input[start + col];

        if (use_shared) {
            values[col] = x;
        }

        maximum = fmaxf(maximum, x);
    }

    maximum = ReduceBlock(maximum, partial, true);

    float sum = 0.0f;

    for (std::size_t col = threadIdx.x; col < columns; col += blockDim.x) {
        float x = use_shared ? values[col] : input[start + col];
        float e = expf(x - maximum);

        if (use_shared) {
            values[col] = e;
        } else {
            output[start + col] = e;
        }

        sum += e;
    }

    sum = ReduceBlock(sum, partial, false);

    float scale = 1.0f / sum;

    for (std::size_t col = threadIdx.x; col < columns; col += blockDim.x) {
        float e = use_shared ? values[col] : output[start + col];
        output[start + col] = e * scale;
    }
}

}  // namespace

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (input.empty()) {
        return {};
    }

    std::size_t columns = input.size() / row_count;
    std::size_t bytes = input.size() * sizeof(float);

    static thread_local GpuBuffer buffer;

    if (bytes > buffer.capacity) {
        float* new_data = nullptr;
        Check(cudaMalloc(&new_data, 2 * bytes));

        cudaFree(buffer.data);
        buffer.data = new_data;
        buffer.capacity = bytes;
    }

    float* device_input = buffer.data;
    float* device_output = buffer.data + input.size();

    Check(cudaMemcpy(device_input, input.data(), bytes, cudaMemcpyHostToDevice));

    int threads = 32;
    while (threads < 256 && static_cast<std::size_t>(threads) < columns) {
        threads *= 2;
    }

    std::size_t shared_bytes = columns <= 8192 ? columns * sizeof(float) : 0;

    SoftmaxKernel<<<row_count, threads, shared_bytes>>>(
        device_input, device_output, columns);

    Check(cudaGetLastError());

    std::vector<float> output(input.size());

    Check(cudaMemcpy(output.data(), device_output, bytes, cudaMemcpyDeviceToHost));

    return output;
}
