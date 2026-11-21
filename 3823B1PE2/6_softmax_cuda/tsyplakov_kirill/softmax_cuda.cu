#include "softmax_cuda.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <cmath>

namespace {

constexpr int BLOCK_SIZE = 256;

float *g_d_in = nullptr, *g_d_out = nullptr;
std::size_t g_cap = 0;

void ensureBuffers(std::size_t bytes) {
    if (bytes <= g_cap) return;
    if (g_d_in) cudaFree(g_d_in);
    if (g_d_out) cudaFree(g_d_out);
    cudaMalloc(&g_d_in, bytes);
    cudaMalloc(&g_d_out, bytes);
    g_cap = bytes;
}

__device__ float blockReduceMax(float val, float* shared) {
    int tid = threadIdx.x;
    shared[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }
    return shared[0];
}

__device__ float blockReduceSum(float val, float* shared) {
    int tid = threadIdx.x;
    shared[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    return shared[0];
}

__global__ void SoftmaxKernel(const float* __restrict__ in,
                              float* __restrict__ out,
                              int row_size) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* row_in  = in  + (std::size_t)row * row_size;
    float*       row_out = out + (std::size_t)row * row_size;

    extern __shared__ float smem[];
    float* reduce_buf = smem;
    float* row_cache  = smem + BLOCK_SIZE;

    float local_max = -INFINITY;
    for (int i = tid; i < row_size; i += blockDim.x) {
        float v = row_in[i];
        row_cache[i] = v;
        local_max = fmaxf(local_max, v);
    }
    float row_max = blockReduceMax(local_max, reduce_buf);

    float local_sum = 0.0f;
    for (int i = tid; i < row_size; i += blockDim.x) {
        float e = __expf(row_cache[i] - row_max);
        row_cache[i] = e;
        local_sum += e;
    }
    float row_sum = blockReduceSum(local_sum, reduce_buf);

    float inv_sum = 1.0f / row_sum;
    for (int i = tid; i < row_size; i += blockDim.x) {
        row_out[i] = row_cache[i] * inv_sum;
    }
}

} // namespace

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    std::size_t total = input.size();
    std::vector<float> output(total);
    if (total == 0 || row_count <= 0) return output;

    int row_size = static_cast<int>(total / row_count);
    std::size_t bytes = total * sizeof(float);

    ensureBuffers(bytes);
    cudaMemcpy(g_d_in, input.data(), bytes, cudaMemcpyHostToDevice);

    std::size_t smem_bytes = (BLOCK_SIZE + row_size) * sizeof(float);

    SoftmaxKernel<<<row_count, BLOCK_SIZE, smem_bytes>>>(g_d_in, g_d_out, row_size);

    cudaMemcpy(output.data(), g_d_out, bytes, cudaMemcpyDeviceToHost);
    return output;
}