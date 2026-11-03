#include "gelu_cuda.h"
#include <cuda_runtime.h>

__global__ void gelu_kernel(const float4* __restrict__ in, 
                            float4* __restrict__ out, 
                            int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n4) {
        float4 x = in[i];
        float4 res;
        
        constexpr float kSqrt2OverPi = 0.7978845608f;
        constexpr float kCoef = 0.044715f;

        float arg_x = -2.0f * kSqrt2OverPi * (x.x + kCoef * x.x * x.x * x.x);
        float arg_y = -2.0f * kSqrt2OverPi * (x.y + kCoef * x.y * x.y * x.y);
        float arg_z = -2.0f * kSqrt2OverPi * (x.z + kCoef * x.z * x.z * x.z);
        float arg_w = -2.0f * kSqrt2OverPi * (x.w + kCoef * x.w * x.w * x.w);

        res.x = x.x * __fdividef(1.0f, 1.0f + __expf(arg_x));
        res.y = x.y * __fdividef(1.0f, 1.0f + __expf(arg_y));
        res.z = x.z * __fdividef(1.0f, 1.0f + __expf(arg_z));
        res.w = x.w * __fdividef(1.0f, 1.0f + __expf(arg_w));

        out[i] = res;
    }
}

__global__ void gelu_kernel_tail(const float* __restrict__ in, 
                                 float* __restrict__ out, 
                                 int start, int n) {
    int i = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        constexpr float kSqrt2OverPi = 0.7978845608f;
        constexpr float kCoef = 0.044715f;
        float arg = -2.0f * kSqrt2OverPi * (x + kCoef * x * x * x);
        out[i] = x * __fdividef(1.0f, 1.0f + __expf(arg));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const int n = input.size();
    const size_t bytes = n * sizeof(float);
    std::vector<float> result(n);

    static float* d_in = nullptr;
    static float* d_out = nullptr;
    static int cached_n = 0;

    if (cached_n != n) {
        if (d_in) cudaFree(d_in);
        if (d_out) cudaFree(d_out);
        cudaMalloc(&d_in, bytes);
        cudaMalloc(&d_out, bytes);
        cached_n = n;
    }

    cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice);
    
    const int threadsPerBlock = 512;
    int n4 = n / 4;

    if (n4 > 0) {
        int blocks4 = (n4 + threadsPerBlock - 1) / threadsPerBlock;
        gelu_kernel<<<blocks4, threadsPerBlock>>>((float4*)d_in, (float4*)d_out, n4);
    }

    int tail_start = n4 * 4;
    if (tail_start < n) {
        int tail_count = n - tail_start;
        int blocks_tail = (tail_count + threadsPerBlock - 1) / threadsPerBlock;
        gelu_kernel_tail<<<blocks_tail, threadsPerBlock>>>(d_in, d_out, tail_start, n);
    }
    
    cudaMemcpy(result.data(), d_out, bytes, cudaMemcpyDeviceToHost);

    return result;
}