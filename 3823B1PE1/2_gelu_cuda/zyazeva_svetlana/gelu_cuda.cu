#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

constexpr float GELU_COEF = 0.044715f;
constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;

__global__ void GeluKernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float x = input[idx];
        float x_cubed = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + GELU_COEF * x_cubed);
        float sigmoid_approx = 1.0f / (1.0f + expf(-2.0f * inner));
        float tanh_approx = 2.0f * sigmoid_approx - 1.0f;

        output[idx] = 0.5f * x * (1.0f + tanh_approx);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int size = input.size();
    if (size == 0) {
        return std::vector<float>();
    }

    std::vector<float> result(size);

    float *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    cudaMemcpy(d_input, input.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    GeluKernel<<<numBlocks, blockSize>>>(d_input, d_output, size);
    cudaDeviceSynchronize();

    cudaMemcpy(result.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return result;
}
