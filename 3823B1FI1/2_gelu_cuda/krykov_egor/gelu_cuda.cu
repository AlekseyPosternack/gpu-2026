#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <cmath>

struct DeviceBuffers {
    float* in = nullptr;
    float* out = nullptr;
    size_t capacity = 0;

    void Reserve(size_t n) {
        if (n <= capacity) return;
        cudaFree(in);
        cudaFree(out);
        cudaMalloc(&in, n * sizeof(float));
        cudaMalloc(&out, n * sizeof(float));
        capacity = n;
    }
    ~DeviceBuffers() {
        cudaFree(in);
        cudaFree(out);
    }
};

__global__ void GeluKernel(const float* in, float* out, size_t n) {
    const float k = 2.0f * 0.7978845608028654f;
    const float c = 0.044715f;

    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) {
        const float x = in[i];
        const float z = k * (x + c * x * x * x);
        out[i] = x - x / (expf(z) + 1.0f);
    }
}
std::vector<float> GeluCUDA(const std::vector<float>& input) {
    static DeviceBuffers buf;
    const size_t n = input.size();
    if (n == 0) return {};

    buf.Reserve(n);

    cudaMemcpy(buf.in, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    const int block = 256;
    const int grid = static_cast<int>((n + block - 1) / block);
    GeluKernel<<<grid, block>>>(buf.in, buf.out, n);

    std::vector<float> result(n);

    cudaMemcpy(result.data(), buf.out, n * sizeof(float), cudaMemcpyDeviceToHost);

    return result;
}