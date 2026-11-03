#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace {

void CheckCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

__global__ void NaiveGemmKernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n || column >= n) {
        return;
    }

    const int rowOffset = row * n;
    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
        sum = fmaf(a[rowOffset + k], b[k * n + column], sum);
    }
    c[rowOffset + column] = sum;
}

class DeviceStorage {
public:
    ~DeviceStorage() {
        cudaFree(data_);
    }

    float* Reserve(std::size_t size) {
        if (size <= capacity_) {
            return data_;
        }
        if (data_ != nullptr) {
            CheckCuda(cudaFree(data_), "cudaFree");
            data_ = nullptr;
            capacity_ = 0;
        }
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&data_), size * sizeof(float)), "cudaMalloc");
        capacity_ = size;
        return data_;
    }

private:
    float* data_ = nullptr;
    std::size_t capacity_ = 0;
};

}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    if (n < 0) {
        throw std::invalid_argument("matrix size must not be negative");
    }
    const std::size_t count = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    if (a.size() != count || b.size() != count) {
        throw std::invalid_argument("matrix dimensions do not match n");
    }
    if (count == 0) {
        return {};
    }

    const std::size_t bytes = count * sizeof(float);
    static thread_local DeviceStorage storage;
    float* deviceA = storage.Reserve(count * 3);
    float* deviceB = deviceA + count;
    float* deviceC = deviceB + count;

    CheckCuda(cudaMemcpy(deviceA, a.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy a");
    CheckCuda(cudaMemcpy(deviceB, b.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy b");

    const dim3 block(32, 8);
    const dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    NaiveGemmKernel<<<grid, block>>>(deviceA, deviceB, deviceC, n);
    CheckCuda(cudaGetLastError(), "NaiveGemmKernel launch");

    std::vector<float> result(count);
    CheckCuda(cudaMemcpy(result.data(), deviceC, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy result");
    return result;
}
