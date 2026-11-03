#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>

namespace {

__global__ void ApplyGelu(float* values, std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        constexpr float kScale = 1.5957691216f;
        constexpr float kCubicCoeff = 0.044715f;
        const float value = values[index];
        const float absolute_value = fabsf(value);
        const float argument =kScale * absolute_value * (1.0f + kCubicCoeff * absolute_value * absolute_value);
        const float exp_value = __expf(-argument);
        const float den = 1.0f + exp_value;

        if (value >= 0.0f) {
            values[index] = value / den;
        } else {
            values[index] = value * exp_value / den;
        }
    }
}

}  // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const std::size_t count = input.size();
    if (input.empty()) {
        return std::vector<float>();
    }

    const std::size_t buffer_size_bytes = count * sizeof(float);
    float* dev_values = nullptr;
    cudaMalloc(&dev_values, buffer_size_bytes);
    cudaMemcpyAsync(dev_values, input.data(), buffer_size_bytes, cudaMemcpyHostToDevice);
    constexpr unsigned int kBlockSize = 256;
    const unsigned int grid_size = static_cast<unsigned int>((count - 1) / kBlockSize + 1);
    ApplyGelu<<<grid_size, kBlockSize>>>(dev_values, count);
    std::vector<float> result(count);

    cudaMemcpy(result.data(), dev_values, buffer_size_bytes, cudaMemcpyDeviceToHost);
    cudaFree(dev_values);
    return result;
}