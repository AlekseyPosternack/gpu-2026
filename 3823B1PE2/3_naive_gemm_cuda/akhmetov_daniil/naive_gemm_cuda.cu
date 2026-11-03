#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstddef>

__global__ void naive_gemm_kernel(const float* a, const float* b, float* c, int n) {
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        
        #pragma unroll(8)
        for (int k = 0; k < n; ++k) {
            float a_val = a[row * n + k];
            
            float4 b_vals = reinterpret_cast<const float4*>(&b[k * n + col])[0];
            
            sum.x += a_val * b_vals.x;
            sum.y += a_val * b_vals.y;
            sum.z += a_val * b_vals.z;
            sum.w += a_val * b_vals.w;
        }
        
        reinterpret_cast<float4*>(&c[row * n + col])[0] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    size_t bytes = (size_t)n * n * sizeof(float);

    static float* d_a = nullptr;
    static float* d_b = nullptr;
    static float* d_c = nullptr;
    static size_t cached_bytes = 0;
    static cudaStream_t stream;
    static bool initialized = false;

    if (!initialized) {
        cudaStreamCreate(&stream);
        initialized = true;
    }

    if (cached_bytes < bytes) {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);
        cached_bytes = bytes;
    }

    cudaMemcpyAsync(d_a, a.data(), bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, b.data(), bytes, cudaMemcpyHostToDevice, stream);

    dim3 block(32, 8);
    dim3 grid((n + 127) / 128, (n + 7) / 8);

    naive_gemm_kernel<<<grid, block, 0, stream>>>(d_a, d_b, d_c, n);

    std::vector<float> c_out((size_t)n * n);
    
    cudaMemcpyAsync(c_out.data(), d_c, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    return c_out;
}