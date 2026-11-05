#include "gelu_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace {

constexpr int kBlockSize = 256;
constexpr float kSqrt2OverPi = 0.7978845608028654f;
constexpr float kCubicCoefficient = 0.044715f;

void CheckCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

__device__ __forceinline__ float Gelu(float x) {
    const float x3 = x * x * x;
    const float exponent = -2.0f * kSqrt2OverPi * (x + kCubicCoefficient * x3);
    return __fdividef(x, 1.0f + __expf(exponent));
}

__global__ void GeluKernel(float* data, std::size_t size) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index < size) {
        data[index] = Gelu(data[index]);
    }
}

class DeviceBuffer {
public:
    ~DeviceBuffer() {
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

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const std::size_t size = input.size();
    if (size == 0) {
        return {};
    }

    const std::size_t bytes = size * sizeof(float);
    static thread_local DeviceBuffer buffer;
    float* deviceData = buffer.Reserve(size);

    CheckCuda(cudaMemcpy(deviceData, input.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy host to device");

    const unsigned int blockCount = static_cast<unsigned int>((size + kBlockSize - 1) / kBlockSize);
    GeluKernel<<<blockCount, kBlockSize>>>(deviceData, size);
    CheckCuda(cudaGetLastError(), "GeluKernel launch");
    std::vector<float> result(size);
    CheckCuda(cudaMemcpy(result.data(), deviceData, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy device to host");
    return result;
}
