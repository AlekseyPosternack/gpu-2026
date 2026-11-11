#include "gelu_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace {

void CheckCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) {
        void* memory = nullptr;
        CheckCuda(cudaMalloc(&memory, bytes), "cudaMalloc");
        data_ = static_cast<float*>(memory);
    }

    ~DeviceBuffer() {
        cudaFree(data_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    float* data() const { return data_; }

private:
    float* data_ = nullptr;
};

__global__ void GeluKernel(const float* input, float* output, std::size_t count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        const float x = input[i];
        const float argument = 0.7978845608028654f * (x + 0.044715f * x * x * x);
        output[i] = x / (1.0f + expf(-2.0f * argument));
    }
}

} // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    if (input.empty()) {
        return {};
    }

    const std::size_t count = input.size();
    const std::size_t bytes = count * sizeof(float);
    DeviceBuffer device_input(bytes);
    DeviceBuffer device_output(bytes);

    CheckCuda(cudaMemcpy(device_input.data(), input.data(), bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy host to device");

    constexpr unsigned int kBlockSize = 256;
    const unsigned int blocks = static_cast<unsigned int>((count + kBlockSize - 1) / kBlockSize);
    GeluKernel<<<blocks, kBlockSize>>>(device_input.data(), device_output.data(), count);
    CheckCuda(cudaGetLastError(), "GeluKernel launch");

    std::vector<float> result(count);
    CheckCuda(cudaMemcpy(result.data(), device_output.data(), bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy device to host");
    return result;
}
