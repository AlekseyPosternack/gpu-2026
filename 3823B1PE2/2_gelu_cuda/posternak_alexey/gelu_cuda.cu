#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cstdlib>

__global__ void gelu_kernel(const float* __restrict__ input, 
                            float* __restrict__ output, 
                            size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        float x = input[i];
        float y = 0.7978845608028654f * x * (1.0f + 0.044715f * x * x);
        output[i] = x / (1.0f + expf(-2.0f * y));
    }
}

// device memory
static float* d_input = nullptr;
static float* d_output = nullptr;
static size_t allocated_size = 0;

static void cleanup_cuda_resources() {
    if (d_input) cudaFree(d_input);
    if (d_output) cudaFree(d_output);
    d_input = nullptr;
    d_output = nullptr;
    allocated_size = 0;
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const size_t n = input.size();
    
    if (n == 0) return std::vector<float>();
    
    // allocate and free device memory once
    if (n > allocated_size) {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        
        cudaMalloc(&d_input, n * sizeof(float));
        cudaMalloc(&d_output, n * sizeof(float));
        allocated_size = n;
        
        static bool cleanup_registered = false;
        if (!cleanup_registered) {
            std::atexit(cleanup_cuda_resources);
            cleanup_registered = true;
        }
    }
    
    cudaMemcpyAsync(d_input, input.data(), n * sizeof(float), 
                    cudaMemcpyHostToDevice, 0);
    
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    gelu_kernel<<<blocks, threads>>>(d_input, d_output, n);
    
    // overlap host memory allocation and CUDA computations
    std::vector<float> output(n);
    
    cudaMemcpyAsync(output.data(), d_output, n * sizeof(float), 
                    cudaMemcpyDeviceToHost, 0);

    cudaDeviceSynchronize();
    
    return output;
}