#include "gelu_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {

void CheckCUDA(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

// Each host thread keeps its own reusable buffer for each CUDA device.
struct DeviceBuffer {
    float* data = nullptr;
    std::size_t capacity = 0;
    int device = 0;

    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (data == nullptr) return;
        int previous = 0;
        if (cudaGetDevice(&previous) != cudaSuccess) return;
        if (previous != device && cudaSetDevice(device) != cudaSuccess) return;
        cudaFree(data);
        if (previous != device) cudaSetDevice(previous);
    }

    void Reserve(std::size_t size, int current_device) {
        if (size <= capacity) return;
        float* replacement = nullptr;
        CheckCUDA(cudaMalloc(reinterpret_cast<void**>(&replacement), size * sizeof(float)), "cudaMalloc");
        if (data != nullptr) {
            const cudaError_t status = cudaFree(data);
            if (status != cudaSuccess) {
                cudaFree(replacement);
                CheckCUDA(status, "cudaFree");
            }
        }
        data = replacement;
        capacity = size;
        device = current_device;
    }
};

__global__ void GeluKernel(float* data, std::size_t size) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < size; i += stride) {
        const float x = data[i];
        const float argument = 0.7978845608028654f * (x + 0.044715f * x * x * x);
        // 0.5 * (1 + tanh(a)) = 1 / (1 + exp(-2a)).
        data[i] = x / (1.0f + __expf(-2.0f * argument));
    }
}

} // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    if (input.empty()) return {};

    int device = 0;
    CheckCUDA(cudaGetDevice(&device), "cudaGetDevice");
    thread_local std::unordered_map<int, DeviceBuffer> buffers;
    DeviceBuffer& buffer = buffers[device];
    buffer.Reserve(input.size(), device);

    const std::size_t bytes = input.size() * sizeof(float);
    CheckCUDA(cudaMemcpy(buffer.data, input.data(), bytes, cudaMemcpyHostToDevice), "copy to GPU");

    constexpr unsigned int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>(
        std::min<std::size_t>((input.size() - 1) / threads + 1, 4096));
    GeluKernel<<<blocks, threads>>>(buffer.data, input.size());
    CheckCUDA(cudaGetLastError(), "GeluKernel launch");

    // Host allocation can overlap the asynchronous kernel execution.
    std::vector<float> output(input.size());
    // This blocking copy also waits for the kernel and reports execution errors.
    CheckCUDA(cudaMemcpy(output.data(), buffer.data, bytes, cudaMemcpyDeviceToHost), "copy from GPU");
    return output;
}
