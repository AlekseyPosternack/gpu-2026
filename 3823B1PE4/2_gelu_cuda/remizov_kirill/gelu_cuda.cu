#include "gelu_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <thread>

namespace {

constexpr int kThreads = 256;

// GELU(x) = 0.5 * x * (1 + tanh(z)), z = sqrt(2/pi) * (x + 0.044715 * x^3)
// Using 0.5 * (1 + tanh(z)) == 1 / (1 + exp(-2z)):
//     GELU(x) = x / (1 + exp(-2 * sqrt(2/pi) * (x + 0.044715 * x^3)))
constexpr float kA = -2.0f * 0.7978845608028654f;
constexpr float kB = kA * 0.044715f;

__device__ __forceinline__ float Gelu(float x) {
    return __fdividef(x, 1.0f + __expf(x * (kA + kB * x * x)));
}

// In-place processing: a single device buffer is enough.
// Interaction is vectorized with float4 (4 elements per thread).
__global__ void GeluKernel(float* __restrict__ data, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t n4 = n / 4;
    float4* vec = reinterpret_cast<float4*>(data);

    if (i < n4) {
        float4 v = vec[i];
        v.x = Gelu(v.x);
        v.y = Gelu(v.y);
        v.z = Gelu(v.z);
        v.w = Gelu(v.w);
        vec[i] = v;
    } else if (i - n4 < n % 4) {
        const std::size_t tail = n4 * 4 + (i - n4);
        data[tail] = Gelu(data[tail]);
    }
}

// Device memory is allocated once and reused across calls.
float* DeviceBuffer(std::size_t count) {
    static float* data = nullptr;
    static std::size_t capacity = 0;
    if (count > capacity) {
        cudaFree(data);
        cudaMalloc(&data, count * sizeof(float));
        capacity = count;
    }
    return data;
}

}  // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const std::size_t n = input.size();
    if (n == 0) {
        return {};
    }
    const std::size_t bytes = n * sizeof(float);

    float* data = DeviceBuffer(n);

    // Host allocation (zero-fill + page faults) overlaps with transfer & compute.
    std::vector<float> output;
    std::thread allocator([&output, n] { output.resize(n); });

    cudaMemcpy(data, input.data(), bytes, cudaMemcpyHostToDevice);

    const std::size_t work = n / 4 + n % 4;
    const unsigned blocks = static_cast<unsigned>((work + kThreads - 1) / kThreads);
    GeluKernel<<<blocks, kThreads>>>(data, n);

    allocator.join();
    cudaMemcpy(output.data(), data, bytes, cudaMemcpyDeviceToHost);
    return output;
}
