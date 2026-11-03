#include "softmax_cuda.h"
#include <cuda_runtime.h>
#include <cmath>

#define THREADS_PER_BLOCK 256

__global__ void softmax_kernel(const float* __restrict__ input, 
                               float* __restrict__ output, 
                               int row_size) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* in_row = input + row * row_size;
    float* out_row = output + row * row_size;

    float m = -INFINITY;
    float d = 0.0f;

    for (int i = tid; i < row_size; i += THREADS_PER_BLOCK) {
        float x = in_row[i];
        float m_new = fmaxf(m, x);

        d = d * __expf(m - m_new) + __expf(x - m_new);
        m = m_new;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        float m_other = __shfl_down_sync(0xffffffff, m, offset);
        float d_other = __shfl_down_sync(0xffffffff, d, offset);
        float m_new = fmaxf(m, m_other);
        d = d * __expf(m - m_new) + d_other * __expf(m_other - m_new);
        m = m_new;
    }

    __shared__ float warp_m[8];
    __shared__ float warp_d[8];
    if (tid % 32 == 0) {
        warp_m[tid / 32] = m;
        warp_d[tid / 32] = d;
    }
    __syncthreads();

    if (tid < 32) {
        float m_val = (tid < 8) ? warp_m[tid] : -INFINITY;
        float d_val = (tid < 8) ? warp_d[tid] : 0.0f;
        
        for (int offset = 4; offset > 0; offset >>= 1) {
            float m_other = __shfl_down_sync(0xffffffff, m_val, offset);
            float d_other = __shfl_down_sync(0xffffffff, d_val, offset);
            float m_new = fmaxf(m_val, m_other);
            d_val = d_val * __expf(m_val - m_new) + d_other * __expf(m_other - m_new);
            m_val = m_new;
        }
        if (tid == 0) {
            warp_m[0] = m_val;
            warp_d[0] = d_val;
        }
    }
    __syncthreads();

    float row_max = warp_m[0];
    float row_sum = warp_d[0];
    float inv_sum = __fdividef(1.0f, row_sum);

    for (int i = tid; i < row_size; i += THREADS_PER_BLOCK) {
        out_row[i] = __expf(in_row[i] - row_max) * inv_sum;
    }
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    int total_elements = input.size();
    int row_size = total_elements / row_count;
    size_t bytes = total_elements * sizeof(float);
    
    std::vector<float> output(total_elements);

    static float* d_input = nullptr;
    static float* d_output = nullptr;
    static cudaStream_t stream = nullptr;
    static size_t cached_bytes = 0;

    if (cached_bytes < bytes) {
        if (d_input) { cudaFree(d_input); cudaFree(d_output); }
        if (stream) cudaStreamDestroy(stream);
        cudaMalloc(&d_input, bytes);
        cudaMalloc(&d_output, bytes);
        cudaStreamCreate(&stream);
        cached_bytes = bytes;
    }

    cudaMemcpyAsync(d_input, input.data(), bytes, cudaMemcpyHostToDevice, stream);
    softmax_kernel<<<row_count, THREADS_PER_BLOCK, 0, stream>>>(d_input, d_output, row_size);
    cudaMemcpyAsync(output.data(), d_output, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    return output;
}