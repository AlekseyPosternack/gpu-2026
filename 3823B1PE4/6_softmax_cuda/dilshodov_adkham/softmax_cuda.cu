#include "softmax_cuda.h"

#include <cuda_runtime.h>

#include <cfloat>

namespace {

constexpr int kMaxBlockSize = 256;
constexpr int kWarpSize = 32;

// Running (max, sum) pair: sum is accumulated relative to max (online softmax),
// so max and sum of exponents are found in a single pass over the row
struct MaxSum {
    float max;
    float sum;
};

__device__ __forceinline__ MaxSum Combine(MaxSum x, MaxSum y) {
    const float m = fmaxf(x.max, y.max);
    if (m == -FLT_MAX) return {m, 0.0f};
    return {m, x.sum * __expf(x.max - m) + y.sum * __expf(y.max - m)};
}

__device__ __forceinline__ MaxSum WarpReduce(MaxSum v) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        MaxSum other;
        other.max = __shfl_xor_sync(0xffffffff, v.max, offset);
        other.sum = __shfl_xor_sync(0xffffffff, v.sum, offset);
        v = Combine(v, other);
    }
    return v;
}

// One CUDA block processes one row
__global__ void SoftmaxKernel(const float* __restrict__ in, float* __restrict__ out, int row_size) {
    __shared__ MaxSum warp_results[kMaxBlockSize / kWarpSize];
    __shared__ MaxSum row_result;

    const float* row_in = in + size_t(blockIdx.x) * row_size;
    float* row_out = out + size_t(blockIdx.x) * row_size;
    const int tid = threadIdx.x;

    // Pass 1: per-thread online max & sum
    MaxSum v = {-FLT_MAX, 0.0f};
    for (int i = tid; i < row_size; i += blockDim.x) {
        const float x = row_in[i];
        if (x > v.max) {
            v.sum = v.sum * __expf(v.max - x) + 1.0f;
            v.max = x;
        } else {
            v.sum += __expf(x - v.max);
        }
    }

    // Block reduction: warp shuffles, then the first warp combines per-warp results
    v = WarpReduce(v);
    const int warp = tid / kWarpSize;
    const int lane = tid % kWarpSize;
    if (lane == 0) warp_results[warp] = v;
    __syncthreads();
    if (warp == 0) {
        const int warp_count = blockDim.x / kWarpSize;
        v = lane < warp_count ? warp_results[lane] : MaxSum{-FLT_MAX, 0.0f};
        v = WarpReduce(v);
        if (lane == 0) row_result = v;
    }
    __syncthreads();

    // Pass 2: normalize (row is read again, mostly from L2 cache)
    const float row_max = row_result.max;
    const float inv_sum = 1.0f / row_result.sum;
    for (int i = tid; i < row_size; i += blockDim.x) {
        row_out[i] = __expf(row_in[i] - row_max) * inv_sum;
    }
}

struct Buffers {
    float* in = nullptr;
    float* out = nullptr;
    size_t capacity = 0;

    void Reserve(size_t count) {
        if (count <= capacity) return;
        cudaFree(in);
        cudaFree(out);
        cudaMalloc(&in, count * sizeof(float));
        cudaMalloc(&out, count * sizeof(float));
        capacity = count;
    }
};

}  // namespace

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    const size_t count = input.size();
    if (count == 0 || row_count <= 0) return std::vector<float>(count);
    const int row_size = static_cast<int>(count / row_count);
    const size_t bytes = count * sizeof(float);

    static Buffers buf;  // device memory is allocated once and reused between calls
    buf.Reserve(count);

    cudaMemcpy(buf.in, input.data(), bytes, cudaMemcpyHostToDevice);

    int block_size = kWarpSize;
    while (block_size < kMaxBlockSize && block_size < row_size) block_size *= 2;
    SoftmaxKernel<<<row_count, block_size>>>(buf.in, buf.out, row_size);

    // Kernel launch is asynchronous, so the result is allocated on host while it runs
    std::vector<float> output(count);
    cudaMemcpy(output.data(), buf.out, bytes, cudaMemcpyDeviceToHost);
    return output;
}
