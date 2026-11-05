#include "gelu_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace {

constexpr int BLOCK_SIZE = 256;

constexpr float SQRT_2_OVER_PI = 0.7978845608f;
constexpr float CUBIC_COEFFICIENT = 0.044715f;

__global__ void GeluKernel(
    const float* input,
    float* output,
    std::size_t n
) {
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx < n) {
        const float x = input[idx];

        const float x2 = x * x;
        const float x3 = x2 * x;

        const float z = SQRT_2_OVER_PI *
                        (x + CUBIC_COEFFICIENT * x3);

        output[idx] = x / (1.0f + expf(-2.0f * z));
    }
}

void CheckCUDA(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(error)
        );
    }
}

} // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const std::size_t n = input.size();

    std::vector<float> output(n);

    if (n == 0) {
        return output;
    }

    const std::size_t bytes = n * sizeof(float);

    float* device_memory = nullptr;

    try {
        CheckCUDA(
            cudaMalloc(
                reinterpret_cast<void**>(&device_memory),
                2 * bytes
            ),
            "cudaMalloc"
        );

        float* device_input = device_memory;
        float* device_output = device_memory + n;

        CheckCUDA(
            cudaMemcpy(
                device_input,
                input.data(),
                bytes,
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy H2D"
        );

        const unsigned int blocks =
            static_cast<unsigned int>(
                (n + BLOCK_SIZE - 1) / BLOCK_SIZE
            );

        GeluKernel<<<blocks, BLOCK_SIZE>>>(
            device_input,
            device_output,
            n
        );

        CheckCUDA(cudaGetLastError(), "GeluKernel launch");

        CheckCUDA(
            cudaMemcpy(
                output.data(),
                device_output,
                bytes,
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy D2H"
        );

        cudaError_t free_error = cudaFree(device_memory);
        device_memory = nullptr;

        CheckCUDA(free_error, "cudaFree");
    }
    catch (...) {
        if (device_memory != nullptr) {
            cudaFree(device_memory);
        }
        throw;
    }

    return output;
}