#include "softmax_cuda.h"
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <algorithm>

#define WARP_SIZE 32

// Warp-level reduction for finding the maximum value using registers
__inline__ __device__ float warpReduceMax(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp-level reduction for summing values using registers
__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction for maximum using shared memory across warps
__inline__ __device__ float blockReduceMax(float val) {
    __shared__ float shared[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warpReduceMax(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    int num_warps = blockDim.x / WARP_SIZE;
    val = (threadIdx.x < num_warps) ? shared[lane] : -FLT_MAX;

    if (wid == 0) {
        val = warpReduceMax(val);
        if (threadIdx.x == 0) {
            shared[0] = val;
        }
    }
    __syncthreads();

    return shared[0];
}

// Block-level reduction for sum using shared memory across warps
__inline__ __device__ float blockReduceSum(float val) {
    __shared__ float shared[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warpReduceSum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    int num_warps = blockDim.x / WARP_SIZE;
    val = (threadIdx.x < num_warps) ? shared[lane] : 0.0f;

    if (wid == 0) {
        val = warpReduceSum(val);
        if (threadIdx.x == 0) {
            shared[0] = val;
        }
    }
    __syncthreads();

    return shared[0];
}

// Kernel 1: Caches the entire row in shared memory (for small to medium rows, e.g., <= 4096)
// Global memory is read only once and written once.
__global__ void softmaxSharedMemKernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int row_size) {
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* row_in = input + static_cast<size_t>(row) * row_size;
    float* row_out = output + static_cast<size_t>(row) * row_size;

    float local_max = -FLT_MAX;

    // 1. Coalesced load from global memory into shared memory & compute thread-local max
    for (int i = tid; i < row_size; i += blockDim.x) {
        float val = row_in[i];
        sdata[i] = val;
        local_max = fmaxf(local_max, val);
    }

    float row_max = (blockDim.x <= WARP_SIZE) ? warpReduceMax(local_max) : blockReduceMax(local_max);
    if (blockDim.x <= WARP_SIZE) {
        row_max = __shfl_sync(0xffffffff, row_max, 0);
    }

    // 2. Compute exponentials in-place in shared memory and accumulate thread-local sum
    float local_sum = 0.0f;
    for (int i = tid; i < row_size; i += blockDim.x) {
        float val = __expf(sdata[i] - row_max);
        sdata[i] = val;
        local_sum += val;
    }

    float row_sum = (blockDim.x <= WARP_SIZE) ? warpReduceSum(local_sum) : blockReduceSum(local_sum);
    if (blockDim.x <= WARP_SIZE) {
        row_sum = __shfl_sync(0xffffffff, row_sum, 0);
    }

    float inv_sum = 1.0f / row_sum;

    // 3. Normalize and write back to global memory
    for (int i = tid; i < row_size; i += blockDim.x) {
        row_out[i] = sdata[i] * inv_sum;
    }
}

// Kernel 2: Vectorized (float4) kernel for large rows (> 4096 elements) that exceed shared memory
__global__ void softmaxLargeRowKernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      int row_size) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float4* row_in4 = reinterpret_cast<const float4*>(input + static_cast<size_t>(row) * row_size);
    float4* row_out4 = reinterpret_cast<float4*>(output + static_cast<size_t>(row) * row_size);
    int vec_size = row_size / 4;

    // 1. Find max using vectorized loads into registers
    float local_max = -FLT_MAX;
    for (int i = tid; i < vec_size; i += blockDim.x) {
        float4 v = row_in4[i];
        local_max = fmaxf(local_max, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
    }
    float row_max = blockReduceMax(local_max);

    // 2. Compute sum of shifted exponentials
    float local_sum = 0.0f;
    for (int i = tid; i < vec_size; i += blockDim.x) {
        float4 v = row_in4[i];
        local_sum += __expf(v.x - row_max) + __expf(v.y - row_max) +
                     __expf(v.z - row_max) + __expf(v.w - row_max);
    }
    float row_sum = blockReduceSum(local_sum);
    float inv_sum = 1.0f / row_sum;

    // 3. Normalize and store
    for (int i = tid; i < vec_size; i += blockDim.x) {
        float4 v = row_in4[i];
        float4 out;
        out.x = __expf(v.x - row_max) * inv_sum;
        out.y = __expf(v.y - row_max) * inv_sum;
        out.z = __expf(v.z - row_max) * inv_sum;
        out.w = __expf(v.w - row_max) * inv_sum;
        row_out4[i] = out;
    }
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (input.empty() || row_count <= 0) {
        return {};
    }

    size_t total_elements = input.size();
    int row_size = static_cast<int>(total_elements / row_count);
    size_t bytes = total_elements * sizeof(float);

    float* d_input = nullptr;
    float* d_output = nullptr;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Allocate device memory
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    // Asynchronously copy input data to device
    cudaMemcpyAsync(d_input, input.data(), bytes, cudaMemcpyHostToDevice, stream);

    // Configure kernel execution parameters
    if (row_size <= 4096) {
        // Choose power-of-2 block size between 32 and 512
        int threads = std::min(512, std::max(32, row_size));
        size_t shared_bytes = row_size * sizeof(float);
        softmaxSharedMemKernel<<<row_count, threads, shared_bytes, stream>>>(
            d_input, d_output, row_size);
    } else {
        int threads = 256;
        softmaxLargeRowKernel<<<row_count, threads, 0, stream>>>(
            d_input, d_output, row_size);
    }

    // Overlap host memory allocation/zero-initialization
    std::vector<float> output(total_elements);

    // Asynchronously copy results back to host and synchronize
    cudaMemcpyAsync(output.data(), d_output, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Cleanup resources
    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream);

    return output;
}