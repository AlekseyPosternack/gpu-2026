#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <thread>

namespace {

__global__ void GeluKernel(float* data, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const float x = data[i];
        const float c1 = 1.5957691216f;
        const float c2 = 0.0713548163f;
        const float u = x * (c1 + c2 * x * x);
        data[i] = x / (1.0f + __expf(-u));
    }
}

struct DeviceBuffer {
    float* ptr = nullptr;
    size_t capacity = 0;

    float* Get(size_t n) {
        if (n > capacity) {
            if (ptr != nullptr) {
                cudaFree(ptr);
            }
            cudaMalloc(&ptr, n * sizeof(float));
            capacity = n;
        }
        return ptr;
    }

    ~DeviceBuffer() {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }
};

DeviceBuffer g_device_buffer;

}  // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const size_t n = input.size();
    if (n == 0) {
        return {};
    }

    const size_t bytes = n * sizeof(float);
    float* d_data = g_device_buffer.Get(n);

    std::vector<float> output;
    std::thread alloc_thread([&output, n] { output.resize(n); });

    cudaMemcpy(d_data, input.data(), bytes, cudaMemcpyHostToDevice);

    const int block_size = 256;
    const size_t num_blocks = (n + block_size - 1) / block_size;
    GeluKernel<<<static_cast<unsigned int>(num_blocks), block_size>>>(d_data, n);

    alloc_thread.join();

    cudaMemcpy(output.data(), d_data, bytes, cudaMemcpyDeviceToHost);

    return output;
}
