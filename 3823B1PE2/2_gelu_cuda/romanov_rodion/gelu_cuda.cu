#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>

__global__ void gelu_kernel(const float* input, float* output, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float x3 = x * x * x;
        const float sqrt_2_pi = 0.7978845608028654f;
        const float coeff = 0.044715f;
        
        float arg = sqrt_2_pi * (x + coeff * x3);
        float exp_val = __expf(2.0f * arg);
        float tanh_approx = (exp_val - 1.0f) / (exp_val + 1.0f);
        
        output[idx] = 0.5f * x * (1.0f + tanh_approx);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int n = input.size();
    std::vector<float> output(n);

    float *d_input = nullptr;
    float *d_output = nullptr;
    size_t bytes = n * sizeof(float);

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    gelu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);

    cudaDeviceSynchronize();

    cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}