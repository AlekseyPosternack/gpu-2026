#include "softmax_cuda.h"
#include <cuda_runtime.h>
#include <cmath>

__global__ void softmax_kernel(const float* input, float* output, int row_size) {
    extern __shared__ float sdata[];

    const float* row_in = input + (size_t)blockIdx.x * row_size;
    float* row_out = output + (size_t)blockIdx.x * row_size;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    float local_max = -INFINITY;
    for (int i = tid; i < row_size; i += nthreads) {
        local_max = fmaxf(local_max, row_in[i]);
    }

    sdata[tid] = local_max;
    __syncthreads();

    for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < row_size; i += nthreads) {
        local_sum += expf(row_in[i] - row_max);
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    float sum = sdata[0];

    for (int i = tid; i < row_size; i += nthreads) {
        row_out[i] = expf(row_in[i] - row_max) / sum;
    }
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    int row_size = input.size() / row_count;
    size_t bytes = input.size() * sizeof(float);

    float* d_in;
    float* d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice);

    int block = 32;
    while (block < 512 && block < row_size) block <<= 1;

    softmax_kernel<<<row_count, block, block * sizeof(float)>>>(d_in, d_out, row_size);

    std::vector<float> result(input.size());
    cudaMemcpy(result.data(), d_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    return result;
}