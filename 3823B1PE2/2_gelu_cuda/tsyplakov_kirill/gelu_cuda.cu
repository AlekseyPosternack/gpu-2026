#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>

__global__ void gelu_kernel(const float* __restrict__ in,
                            float* __restrict__ out,
                            size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
        float e = expf(2.0f * inner);
        float tanh_val = (e - 1.0f) / (e + 1.0f);
        out[i] = 0.5f * x * (1.0f + tanh_val);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t n = input.size();
    std::vector<float> output(n);
    if (n == 0) return output;

    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in,  n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    gelu_kernel<<<blocks, threads>>>(d_in, d_out, n);

    cudaMemcpy(output.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
    return output;
}