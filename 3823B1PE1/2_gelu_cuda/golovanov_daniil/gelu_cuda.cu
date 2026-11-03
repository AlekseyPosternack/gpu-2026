#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace {

constexpr int kThreads = 256;
constexpr float kSqrt2OverPi = 0.7978845608028654f;
constexpr float kAlpha = 0.044715f;

__device__ __forceinline__ float gelu_scalar(float x) {
    const float x3 = x * x * x;
    const float z = kSqrt2OverPi * (x + kAlpha * x3);
    const float e = __expf(2.0f * z);
    const float t = 1.0f - 2.0f / (e + 1.0f);
    return 0.5f * x * (1.0f + t);
}

__global__ void gelu_kernel(const float* __restrict__ in,
                            float* __restrict__ out,
                            std::size_t n) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t idx = i; idx < n; idx += stride) {
        out[idx] = gelu_scalar(in[idx]);
    }
}

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

struct DeviceBuffer {
    float* ptr = nullptr;
    std::size_t cap = 0;

    void ensure(std::size_t n) {
        if (n > cap) {
            if (ptr) check_cuda(cudaFree(ptr), "cudaFree");
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(float)), "cudaMalloc");
            cap = n;
        }
    }

    ~DeviceBuffer() {
        if (ptr) cudaFree(ptr);
    }
};

} // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const std::size_t n = input.size();
    std::vector<float> output(n);
    if (n == 0) return output;

    static DeviceBuffer d_in;
    static DeviceBuffer d_out;

    d_in.ensure(n);
    d_out.ensure(n);

    const std::size_t bytes = n * sizeof(float);

    check_cuda(cudaMemcpyAsync(d_in.ptr, input.data(), bytes,
                               cudaMemcpyHostToDevice, 0),
               "cudaMemcpy H2D");

    const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
    gelu_kernel<<<blocks, kThreads, 0, 0>>>(d_in.ptr, d_out.ptr, n);

    check_cuda(cudaGetLastError(), "gelu_kernel launch");

    check_cuda(cudaMemcpyAsync(output.data(), d_out.ptr, bytes,
                               cudaMemcpyDeviceToHost, 0),
               "cudaMemcpy D2H");

    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    return output;
}