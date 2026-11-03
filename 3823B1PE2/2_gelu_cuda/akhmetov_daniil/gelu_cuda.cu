#include "gelu_cuda.h"
#include <vector>
#include <cstddef>
#include <cuda_runtime.h>

__global__ void gelu_kernel(const float* input, float* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        
        float x2 = x * x;
        float exponent = x * fmaf(x2, -0.07135481f, -1.59576912f);
        
        output[idx] = x / (1.0f + __expf(exponent));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t size = input.size();
    size_t bytes = size * sizeof(float);

    static float* d_in = nullptr;
    static float* d_out = nullptr;
    static size_t cached_size = 0;
    static cudaStream_t stream;
    static bool initialized = false;

    if (!initialized) {
        cudaStreamCreate(&stream);
        initialized = true;
    }

    if (cached_size < size) {
        if (d_in) cudaFree(d_in);
        if (d_out) cudaFree(d_out);
        cudaMalloc(&d_in, bytes);
        cudaMalloc(&d_out, bytes);
        cached_size = size;
    }

    cudaMemcpyAsync(d_in, input.data(), bytes, cudaMemcpyHostToDevice, stream);

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    gelu_kernel<<<numBlocks, blockSize, 0, stream>>>(d_in, d_out, size);

    std::vector<float> output(size);

    cudaMemcpyAsync(output.data(), d_out, bytes, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    return output;
}