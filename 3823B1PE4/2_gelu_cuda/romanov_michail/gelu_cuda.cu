#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <thread>

namespace {

constexpr int BLOCK_SIZE = 256;

constexpr float COEFF_A = -2.0f * 0.7978845608028654f;
constexpr float COEFF_B = COEFF_A * 0.044715f;

__device__ __forceinline__ float FastGelu(float x) {
    return __fdividef(x, 1.0f + __expf(x * (COEFF_A + COEFF_B * x * x)));
}

__global__ void GeluKernel(float* __restrict__ data, std::size_t size) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t vec_size = size / 4;
    
    float4* vec_ptr = reinterpret_cast<float4*>(data);

    if (idx < vec_size) {
        float4 v = vec_ptr[idx];
        v.x = FastGelu(v.x);
        v.y = FastGelu(v.y);
        v.z = FastGelu(v.z);
        v.w = FastGelu(v.w);
        vec_ptr[idx] = v;
    } else {
        const std::size_t tail_idx = vec_size * 4 + (idx - vec_size);
        if (tail_idx < size) {
            data[tail_idx] = FastGelu(data[tail_idx]);
        }
    }
}

float* GetDeviceBuffer(std::size_t count) {
    static float* dev_ptr = nullptr;
    static std::size_t current_capacity = 0;

    if (count > current_capacity) {
        if (dev_ptr != nullptr) {
            cudaFree(dev_ptr);
        }
        cudaMalloc(&dev_ptr, count * sizeof(float));
        current_capacity = count;
    }
    return dev_ptr;
}

} // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const std::size_t size = input.size();
    if (size == 0) {
        return {};
    }

    const std::size_t bytes = size * sizeof(float);
    float* d_data = GetDeviceBuffer(size);

    std::vector<float> output;
    std::thread host_alloc_thread([&output, size] {
        output.resize(size);
    });

    cudaMemcpy(d_data, input.data(), bytes, cudaMemcpyHostToDevice);

    const std::size_t total_threads = size / 4 + (size % 4);
    const unsigned blocks = static_cast<unsigned>((total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE);

    GeluKernel<<<blocks, BLOCK_SIZE>>>(d_data, size);

    host_alloc_thread.join();

    cudaMemcpy(output.data(), d_data, bytes, cudaMemcpyDeviceToHost);

    return output;
}