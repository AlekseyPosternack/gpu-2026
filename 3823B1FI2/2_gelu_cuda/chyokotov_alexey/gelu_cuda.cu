#include "gelu_cuda.h"
#include <cuda_runtime.h>

__global__ void gelu_kernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float SQRT_2_OVER_PI = 0.7978845608f;
    const float COEFF = 0.044715f;

    float x = in[i];
    float arg = SQRT_2_OVER_PI * (x + COEFF * x * x * x);

    float e = __expf(-2.0f * arg);
    float tanh_approx = (1.0f - e) / (1.0f + e);

    out[i] = 0.5f * x * (1.0f + tanh_approx);
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t size = input.size();
    std::vector<float> output(size);
    if (size == 0) return output;

    size_t bytes = size * sizeof(float);
    float *d_in, *d_out;

    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    gelu_kernel<<<blocks, threads>>>(d_in, d_out, size);

    cudaMemcpy(output.data(), d_out, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);

    return output;
}