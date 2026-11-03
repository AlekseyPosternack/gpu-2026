#include "softmax_cuda.h"

#include <cfloat>

#include <cuda_runtime.h>

namespace {

constexpr int kWarpSize = 32;
constexpr int kBlockSize = 256;
constexpr int kWarpsPerBlock = kBlockSize / kWarpSize;

__device__ __forceinline__ float WarpMax(float v) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
    }
    return v;
}

__device__ __forceinline__ float WarpSum(float v) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, offset);
    }
    return v;
}

template <int kPerLane>
__global__ void SoftmaxWarpKernel(const float* __restrict__ in, float* __restrict__ out,
                                  int row_count, int row_size) {
    const int lane = threadIdx.x % kWarpSize;
    const int row = blockIdx.x * kWarpsPerBlock + threadIdx.x / kWarpSize;
    if (row >= row_count) {
        return;
    }
    const float* src = in + static_cast<size_t>(row) * row_size;
    float* dst = out + static_cast<size_t>(row) * row_size;

    float vals[kPerLane];
    float max_val = -FLT_MAX;
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
        const int idx = lane + i * kWarpSize;
        vals[i] = idx < row_size ? src[idx] : -FLT_MAX;
        max_val = fmaxf(max_val, vals[i]);
    }
    max_val = WarpMax(max_val);

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
        const int idx = lane + i * kWarpSize;
        vals[i] = idx < row_size ? __expf(vals[i] - max_val) : 0.0f;
        sum += vals[i];
    }
    const float inv_sum = 1.0f / WarpSum(sum);

#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
        const int idx = lane + i * kWarpSize;
        if (idx < row_size) {
            dst[idx] = vals[i] * inv_sum;
        }
    }
}

__device__ float BlockReduce(float v, bool is_max, float* shared) {
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    v = is_max ? WarpMax(v) : WarpSum(v);
    if (lane == 0) {
        shared[warp] = v;
    }
    __syncthreads();
    if (warp == 0) {
        v = lane < kWarpsPerBlock ? shared[lane] : (is_max ? -FLT_MAX : 0.0f);
        v = is_max ? WarpMax(v) : WarpSum(v);
        if (lane == 0) {
            shared[0] = v;
        }
    }
    __syncthreads();
    const float result = shared[0];
    __syncthreads();
    return result;
}

__global__ void SoftmaxBlockKernel(const float* __restrict__ in, float* __restrict__ out,
                                   int row_size) {
    __shared__ float shared[kWarpsPerBlock];
    const float* src = in + static_cast<size_t>(blockIdx.x) * row_size;
    float* dst = out + static_cast<size_t>(blockIdx.x) * row_size;

    float local_max = -FLT_MAX;
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += kBlockSize) {
        const float x = src[i];
        if (x > local_max) {
            local_sum = local_sum * __expf(local_max - x) + 1.0f;
            local_max = x;
        } else {
            local_sum += __expf(x - local_max);
        }
    }

    const float max_val = BlockReduce(local_max, true, shared);
    local_sum = local_max == -FLT_MAX ? 0.0f : local_sum * __expf(local_max - max_val);
    const float inv_sum = 1.0f / BlockReduce(local_sum, false, shared);

    for (int i = threadIdx.x; i < row_size; i += kBlockSize) {
        dst[i] = __expf(src[i] - max_val) * inv_sum;
    }
}

}  // namespace

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    const size_t size = input.size();
    if (size == 0 || row_count <= 0) {
        return {};
    }
    const int row_size = static_cast<int>(size / row_count);
    const size_t bytes = size * sizeof(float);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float* d_in = nullptr;
    float* d_out = nullptr;
    cudaMallocAsync(&d_in, bytes, stream);
    cudaMallocAsync(&d_out, bytes, stream);
    cudaMemcpyAsync(d_in, input.data(), bytes, cudaMemcpyHostToDevice, stream);

    const int warp_grid = (row_count + kWarpsPerBlock - 1) / kWarpsPerBlock;
    if (row_size <= 32) {
        SoftmaxWarpKernel<1><<<warp_grid, kBlockSize, 0, stream>>>(d_in, d_out, row_count, row_size);
    } else if (row_size <= 128) {
        SoftmaxWarpKernel<4><<<warp_grid, kBlockSize, 0, stream>>>(d_in, d_out, row_count, row_size);
    } else if (row_size <= 512) {
        SoftmaxWarpKernel<16><<<warp_grid, kBlockSize, 0, stream>>>(d_in, d_out, row_count, row_size);
    } else if (row_size <= 1024) {
        SoftmaxWarpKernel<32><<<warp_grid, kBlockSize, 0, stream>>>(d_in, d_out, row_count, row_size);
    } else {
        SoftmaxBlockKernel<<<row_count, kBlockSize, 0, stream>>>(d_in, d_out, row_size);
    }

    std::vector<float> output(size);
    cudaMemcpyAsync(output.data(), d_out, bytes, cudaMemcpyDeviceToHost, stream);

    cudaFreeAsync(d_in, stream);
    cudaFreeAsync(d_out, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    return output;
}
