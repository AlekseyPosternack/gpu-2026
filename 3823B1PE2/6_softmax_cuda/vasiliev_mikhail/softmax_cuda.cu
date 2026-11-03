#include "softmax_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace {

constexpr size_t kMaxCacheFloats = 8192;

__device__ __forceinline__ float WarpMax(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

__device__ __forceinline__ float WarpSum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ float BlockMax(float v, float* red) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = WarpMax(v);
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    v = (lane < (blockDim.x >> 5)) ? red[lane] : -INFINITY;
    return WarpMax(v);
}

__device__ float BlockSum(float v, float* red) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = WarpSum(v);
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    v = (lane < (blockDim.x >> 5)) ? red[lane] : 0.f;
    return WarpSum(v);
}

template <bool kCache>
__global__ void SoftmaxKernel(const float* __restrict__ in,
                              float* __restrict__ out, int row_size) {
    extern __shared__ float row[];
    __shared__ float red[32];

    const float* x = in + static_cast<size_t>(blockIdx.x) * row_size;
    float* y = out + static_cast<size_t>(blockIdx.x) * row_size;

    float m = -INFINITY;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        const float v = x[i];
        if (kCache) row[i] = v;
        m = fmaxf(m, v);
    }
    m = BlockMax(m, red);

    float s = 0.f;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        const float e = __expf((kCache ? row[i] : x[i]) - m);
        if (kCache) row[i] = e;
        s += e;
    }
    s = BlockSum(s, red);

    const float inv = 1.0f / s;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        const float e = kCache ? row[i] : __expf(x[i] - m);
        y[i] = e * inv;
    }
}

float* d_in = nullptr;
float* d_out = nullptr;
size_t capacity = 0;

void Reserve(size_t elems) {
    if (elems <= capacity) return;
    cudaDeviceSynchronize();
    if (d_in) cudaFree(d_in);
    if (d_out) cudaFree(d_out);
    cudaMalloc(&d_in, elems * sizeof(float));
    cudaMalloc(&d_out, elems * sizeof(float));
    capacity = elems;
}

}  // namespace

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (input.empty() || row_count <= 0) return {};

    const size_t elems = input.size();
    const size_t bytes = elems * sizeof(float);
    const int row_size = static_cast<int>(elems / row_count);

    Reserve(elems);
    cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice);

    int block = 32;
    while (block < 512 && block < row_size) block <<= 1;

    if (static_cast<size_t>(row_size) <= kMaxCacheFloats) {
        SoftmaxKernel<true><<<row_count, block, row_size * sizeof(float)>>>(
            d_in, d_out, row_size);
    } else {
        SoftmaxKernel<false><<<row_count, block>>>(d_in, d_out, row_size);
    }

    std::vector<float> output(elems);

    cudaMemcpy(output.data(), d_out, bytes, cudaMemcpyDeviceToHost);
    return output;
}
