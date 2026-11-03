#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <cstddef>

__device__ __forceinline__ float gelu_device(float x) {
    constexpr float kNegTwoSqrt2OverPi = -1.5957691216057308f;
    constexpr float kCoef = 0.044715f;

    const float poly = fmaf(kCoef, x * x, 1.0f);
    const float arg = kNegTwoSqrt2OverPi * x * poly;
    return x / (1.0f + expf(arg));
}

__global__ void gelu_vec4_kernel(const float4* __restrict__ in, float4* __restrict__ out, int n4) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n4) return;

    const float4 v = in[idx];
    float4 r;
    r.x = gelu_device(v.x);
    r.y = gelu_device(v.y);
    r.z = gelu_device(v.z);
    r.w = gelu_device(v.w);
    out[idx] = r;
}

__global__ void gelu_tail_kernel(const float* __restrict__ in, float* __restrict__ out, int offset, int n) {
    const int idx = offset + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = gelu_device(in[idx]);
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const std::size_t n = input.size();
    std::vector<float> output(n);
    if (n == 0) return output;

    const std::size_t bytes = n * sizeof(float);
    float* d_in = nullptr;
    float* d_out = nullptr;

    if (cudaMalloc(&d_in, bytes) != cudaSuccess) {
        return {};
    }
    if (cudaMalloc(&d_out, bytes) != cudaSuccess) {
        cudaFree(d_in);
        return {};
    }

    if (cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_in);
        cudaFree(d_out);
        return {};
    }

    const int n4 = static_cast<int>(n / 4);
    const int tail = static_cast<int>(n % 4);
    const int block = 256;

    if (n4 > 0) {
        const int grid = (n4 + block - 1) / block;
        gelu_vec4_kernel << <grid, block >> > (
            reinterpret_cast<const float4*>(d_in),
            reinterpret_cast<float4*>(d_out),
            n4);
    }

    if (tail > 0) {
        gelu_tail_kernel << <1, 32 >> > (d_in, d_out, n4 * 4, static_cast<int>(n));
    }

    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(d_in);
        cudaFree(d_out);
        return {};
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        cudaFree(d_in);
        cudaFree(d_out);
        return {};
    }

    if (cudaMemcpy(output.data(), d_out, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaFree(d_in);
        cudaFree(d_out);
        return {};
    }

    cudaFree(d_in);
    cudaFree(d_out);

    return output;
}