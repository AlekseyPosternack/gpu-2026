#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <vector>

namespace {

constexpr float kSqrtTwoOverPi = 0.7978845608f;
constexpr float kGeluCoeff = 0.044715f;

__global__ void GeluKernel(const float* input,
                           float* output,
                           int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float x = input[idx];

        float t = kSqrtTwoOverPi * (x + kGeluCoeff * x * x * x);

        float e = __expf(2.0f * t);
        float tanh_t = (e - 1.0f) / (e + 1.0f);

        output[idx] = 0.5f * x * (1.0f + tanh_t);
    }
}

struct DeviceBuffer {
    float* pointer = nullptr;
    int capacity = 0;
};

float* GetBuffer(DeviceBuffer& buffer, int required_size) {
    if (required_size > buffer.capacity) {
        if (buffer.pointer != nullptr) {
            cudaFree(buffer.pointer);
        }

        cudaMalloc(&buffer.pointer, required_size * sizeof(float));
        buffer.capacity = required_size;
    }

    return buffer.pointer;
}

DeviceBuffer input_buffer;
DeviceBuffer output_buffer;

} // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int size = static_cast<int>(input.size());
    std::vector<float> output(size);

    if (size == 0) {
        return output;
    }

    float* d_input = GetBuffer(input_buffer, size);
    float* d_output = GetBuffer(output_buffer, size);

    cudaMemcpy(d_input, input.data(),
               size * sizeof(float),
               cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    GeluKernel<<<grid_size, block_size>>>(d_input, d_output, size);

    cudaMemcpy(output.data(), d_output,
               size * sizeof(float),
               cudaMemcpyDeviceToHost);

    return output;
}
