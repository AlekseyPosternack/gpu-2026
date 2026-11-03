#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <stdexcept>

__device__ __forceinline__ float gelu_func(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
    return x / (1.0f + expf(-2.0f * inner));
}

__global__ void gelu_fast_kernel(const float4* __restrict__ input, float4* __restrict__ output, size_t n_v4, size_t v4_offset) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_v4) {
        float4 in_val = input[v4_offset + idx];
        float4 out_val;
        out_val.x = gelu_func(in_val.x);
        out_val.y = gelu_func(in_val.y);
        out_val.z = gelu_func(in_val.z);
        out_val.w = gelu_func(in_val.w);
        output[v4_offset + idx] = out_val;
    }
}

__global__ void gelu_scalar_tail_kernel(const float* __restrict__ input, float* __restrict__ output, size_t start_idx, size_t end_idx) {
    size_t idx = start_idx + (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < end_idx) {
        output[idx] = gelu_func(input[idx]);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t n = input.size();
    if (n == 0) return {};

    std::vector<float> output(n);
    size_t bytes = n * sizeof(float);

    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    const int NUM_STREAMS = 4;
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    size_t chunkSize = (n + NUM_STREAMS - 1) / NUM_STREAMS;
    chunkSize = ((chunkSize + 3) / 4) * 4;
    int threadsPerBlock = 256;

    for (int i = 0; i < NUM_STREAMS; ++i) {
        size_t offset = i * chunkSize;
        if (offset >= n) break;
        size_t cur_n = std::min(chunkSize, n - offset);
        size_t cur_bytes = cur_n * sizeof(float);

        cudaMemcpyAsync(d_in + offset, input.data() + offset, cur_bytes, cudaMemcpyHostToDevice, streams[i]);

        size_t cur_v4 = cur_n / 4;
        size_t rem = cur_n % 4;
        size_t v4_offset = offset / 4;

        if (cur_v4 > 0) {
            int blocks = static_cast<int>((cur_v4 + threadsPerBlock - 1) / threadsPerBlock);
            gelu_fast_kernel<<<blocks, threadsPerBlock, 0, streams[i]>>>(
                reinterpret_cast<const float4*>(d_in),
                reinterpret_cast<float4*>(d_out),
                cur_v4,
                v4_offset
            );
        }

        if (rem > 0) {
            size_t tail_start = offset + (cur_v4 * 4);
            int blocks = static_cast<int>((rem + threadsPerBlock - 1) / threadsPerBlock);
            gelu_scalar_tail_kernel<<<blocks, threadsPerBlock, 0, streams[i]>>>(
                d_in,
                d_out,
                tail_start,
                offset + cur_n
            );
        }

        cudaMemcpyAsync(output.data() + offset, d_out + offset, cur_bytes, cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaFree(d_in);
    cudaFree(d_out);

    return output;
}