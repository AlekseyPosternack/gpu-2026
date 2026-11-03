
#include "softmax_cuda.h"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <cstddef>
#include <future>
#include <stdexcept>
#include <vector>

__global__ void SoftmaxKernel(float* data, size_t rowSize) {
    __shared__ float shared[256];

    int tid = threadIdx.x;
    size_t base = (size_t)blockIdx.x * rowSize;

    float localMax = -CUDART_INF_F;

    for (size_t i = tid; i < rowSize; i += blockDim.x)
        localMax = fmaxf(localMax, data[base + i]);

    shared[tid] = localMax;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        __syncthreads();
    }

    float maxValue = shared[0];
    float localSum = 0.0f;

    for (size_t i = tid; i < rowSize; i += blockDim.x) {
        float value = expf(data[base + i] - maxValue);
        data[base + i] = value;
        localSum += value;
    }

    shared[tid] = localSum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            shared[tid] += shared[tid + s];
        __syncthreads();
    }

    float invSum = 1.0f / shared[0];

    for (size_t i = tid; i < rowSize; i += blockDim.x)
        data[base + i] *= invSum;
}

std::vector<float> SoftmaxCUDA(
    const std::vector<float>& input,
    int row_count
) {
    if (row_count <= 0)
        throw std::invalid_argument("Invalid row count");

    if (input.size() % (size_t)row_count != 0)
        throw std::invalid_argument("Invalid matrix size");

    if (input.empty()) return {};

    size_t rowSize = input.size() / row_count;
    size_t bytes = input.size() * sizeof(float);

    float* d_data = nullptr;
    std::future<void> copy_future;

    auto check = [](cudaError_t error) {
        if (error != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(error));
        };

    try {
        check(cudaMalloc((void**)&d_data, bytes));

        // Копирование CPU -> GPU в отдельном потоке
        copy_future = std::async(std::launch::async, [&]() {
            check(cudaMemcpy(
                d_data, input.data(), bytes,
                cudaMemcpyHostToDevice
            ));
            check(cudaStreamSynchronize(0));
            });

        // Создание результата параллельно с копированием
        std::vector<float> result(input.size());

        copy_future.get();

        SoftmaxKernel << <row_count, 256 >> > (d_data, rowSize);
        check(cudaGetLastError());

        check(cudaMemcpy(
            result.data(), d_data, bytes,
            cudaMemcpyDeviceToHost
        ));

        check(cudaFree(d_data));
        d_data = nullptr;

        return result;
    }
    catch (...) {
        if (copy_future.valid())
            copy_future.wait();

        if (d_data)
            cudaFree(d_data);

        throw;
    }
}
