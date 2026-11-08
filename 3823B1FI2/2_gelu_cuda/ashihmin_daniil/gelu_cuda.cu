#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>

__global__ void GeluKernel(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        constexpr float A = 1.5957691216057307f;
        constexpr float B = 0.0713548162726009f;

        float exp_arg = -x * (A + B * x * x);
        out[idx] = x / (1.0f + expf(exp_arg));
    }
}

struct CudaMemoryManager {
    float* d_in = nullptr;
    float* d_out = nullptr;
    size_t capacity = 0;

    void ensure_capacity(size_t n) {
        if (n > capacity) {
            if (d_in) cudaFree(d_in);
            if (d_out) cudaFree(d_out);
            cudaMalloc(&d_in, n * sizeof(float));
            cudaMalloc(&d_out, n * sizeof(float));
            capacity = n;
        }
    }

    ~CudaMemoryManager() {
        if (d_in) cudaFree(d_in);
        if (d_out) cudaFree(d_out);
    }
};

static CudaMemoryManager g_cuda_mem;

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const size_t n = input.size();
    if (n == 0) {
        return {};
    }

    const size_t bytes = n * sizeof(float);

    g_cuda_mem.ensure_capacity(n);

    cudaMemcpy(g_cuda_mem.d_in, input.data(), bytes, cudaMemcpyHostToDevice);

    const int threads_per_block = 256;
    const int blocks = static_cast<int>((n + threads_per_block - 1) / threads_per_block);

    GeluKernel<<<blocks, threads_per_block>>>(g_cuda_mem.d_in, g_cuda_mem.d_out, n);


    std::vector<float> output(n);

    cudaMemcpy(output.data(), g_cuda_mem.d_out, bytes, cudaMemcpyDeviceToHost);

    return output;
}