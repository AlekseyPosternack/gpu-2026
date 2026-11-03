#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>

// Precomputed constants:
// k0 = -2.0f * sqrt(2.0f / pi)
// k1 = 0.044715f
#define GELU_K0 -1.595769122f
#define GELU_K1 0.044715f

__global__ void gelu_kernel_in_place(float* __restrict__ d_data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = d_data[idx];
        // Fast hardware exponential
        float exp_val = __expf(GELU_K0 * x * (1.0f + GELU_K1 * x * x));
        d_data[idx] = x / (1.0f + exp_val);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t n = input.size();
    if (n == 0) {
        return {};
    }

    size_t bytes = n * sizeof(float);
    float* d_data = nullptr;

    // Allocate and free device memory once (in-place execution uses 1 buffer instead of 2)
    cudaMalloc(&d_data, bytes);

    // Copy input data to device asynchronously
    cudaMemcpyAsync(d_data, input.data(), bytes, cudaMemcpyHostToDevice);

    // Launch the kernel asynchronously
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    gelu_kernel_in_place<<<blocksPerGrid, threadsPerBlock>>>(d_data, n);

    // Overlap host memory allocation and zero-initialization with GPU execution
    std::vector<float> output(n);

    // Copy results back to host (implicitly synchronizes with the default stream)
    cudaMemcpy(output.data(), d_data, bytes, cudaMemcpyDeviceToHost);

    // Free the single device buffer
    cudaFree(d_data);

    return output;
}