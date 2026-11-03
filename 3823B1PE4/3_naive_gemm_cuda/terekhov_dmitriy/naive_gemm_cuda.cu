#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>

namespace {

void checkCuda(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

__global__ void multiplyKernel(const float* a, const float* b, float* c, int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n) {
        const std::size_t row_start = static_cast<std::size_t>(row) * n;
        float sum = 0.0f;
        #pragma unroll 4
        for (int k = 0; k < n; ++k) {
            sum += a[row_start + k] * b[static_cast<std::size_t>(k) * n + col];
        }
        c[row_start + col] = sum;
    }
}

}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                               const std::vector<float>& b,
                               int n) {
    if (n < 0) {
        throw std::invalid_argument("Matrix size must be nonnegative");
    }
    const std::size_t size = static_cast<std::size_t>(n) * n;
    if (a.size() != size || b.size() != size) {
        throw std::invalid_argument("Matrix dimensions do not match the data");
    }
    if (n == 0) {
        return {};
    }

    const std::size_t bytes = size * sizeof(float);
    float* device_data = nullptr;
    checkCuda(cudaMalloc(&device_data, 3 * bytes));
    float* device_a = device_data;
    float* device_b = device_data + size;
    float* device_c = device_data + 2 * size;

    std::vector<float> result;
    try {
        checkCuda(cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice));

        const dim3 threads(32, 8);
        const dim3 blocks((n - 1) / threads.x + 1, (n - 1) / threads.y + 1);
        multiplyKernel<<<blocks, threads>>>(device_a, device_b, device_c, n);
        checkCuda(cudaGetLastError());

        result.resize(size);
        checkCuda(cudaMemcpy(result.data(), device_c, bytes, cudaMemcpyDeviceToHost));
    } catch (...) {
        cudaFree(device_data);
        throw;
    }

    checkCuda(cudaFree(device_data));
    return result;
}
