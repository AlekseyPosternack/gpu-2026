#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

//Ядро CUDA
__global__ void gelu_kernel(const float* __restrict__ input, float* __restrict__ output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float x = input[i];
        //Оптимизированная формула:0,5*x*(1+tanh(z)) == x/(1+exp(-2z))
        float z = 0.7978845608028654f * (x + 0.044715f * x * x * x);
        output[i] = x / (1.0f + expf(-2.0f * z));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const int size = static_cast<int>(input.size());
    std::vector<float> output(size);

    if (size == 0) return output;

    const size_t bytes = size * sizeof(float);
    float *d_input = nullptr;
    float *d_output = nullptr;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    gelu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, size);
    cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}