#include "softmax_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <mutex>
#include <algorithm>

static float* d_in_cache = nullptr;
static float* d_out_cache = nullptr;
static size_t cached_bytes = 0;
static cudaStream_t stream = nullptr;
static bool initialized = false;
static std::mutex mtx;

constexpr int BLOCK_THREADS = 256;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = BLOCK_THREADS / WARP_SIZE;

__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void __launch_bounds__(BLOCK_THREADS) softmax_final_kernel(const float* __restrict__ input, float* __restrict__ output, int cols) {
    __shared__ float s_reduce_max[WARPS_PER_BLOCK];
    __shared__ float s_reduce_sum[WARPS_PER_BLOCK];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;

    const float4* row_in_v = reinterpret_cast<const float4*>(input + row * cols);
    float4* row_out_v = reinterpret_cast<float4*>(output + row * cols);

    int cols_v = cols / 4;

    float local_max = -INFINITY;
    for (int i = tid; i < cols_v; i += BLOCK_THREADS) {
        float4 val = row_in_v[i];
        local_max = fmaxf(local_max, fmaxf(fmaxf(val.x, val.y), fmaxf(val.z, val.w)));
    }

    for (int i = cols_v * 4 + tid; i < cols; i += BLOCK_THREADS) {
        local_max = fmaxf(local_max, input[row * cols + i]);
    }

    float warp_max = warpReduceMax(local_max);
    if (lane == 0) s_reduce_max[wid] = warp_max;
    __syncthreads();

    float row_max = -INFINITY;
    if (wid == 0) {
        float val = (tid < WARPS_PER_BLOCK) ? s_reduce_max[lane] : -INFINITY;
        row_max = warpReduceMax(val);
    }
    if (tid == 0) s_reduce_max[0] = row_max;
    __syncthreads();
    row_max = s_reduce_max[0];

    float local_sum = 0.0f;
    for (int i = tid; i < cols_v; i += BLOCK_THREADS) {
        float4 val = row_in_v[i];
        local_sum += __expf(val.x - row_max) + __expf(val.y - row_max) + 
                     __expf(val.z - row_max) + __expf(val.w - row_max);
    }
    for (int i = cols_v * 4 + tid; i < cols; i += BLOCK_THREADS) {
        local_sum += __expf(input[row * cols + i] - row_max);
    }

    float warp_sum = warpReduceSum(local_sum);
    if (lane == 0) s_reduce_sum[wid] = warp_sum;
    __syncthreads();

    float row_sum = 0.0f;
    if (wid == 0) {
        float val = (tid < WARPS_PER_BLOCK) ? s_reduce_sum[lane] : 0.0f;
        row_sum = warpReduceSum(val);
    }
    if (tid == 0) s_reduce_sum[0] = row_sum;
    __syncthreads();
    row_sum = s_reduce_sum[0];
    
    float inv_sum = __fdividef(1.0f, row_sum);

    for (int i = tid; i < cols_v; i += BLOCK_THREADS) {
        float4 val = row_in_v[i];
        float4 res;
        res.x = __expf(val.x - row_max) * inv_sum;
        res.y = __expf(val.y - row_max) * inv_sum;
        res.z = __expf(val.z - row_max) * inv_sum;
        res.w = __expf(val.w - row_max) * inv_sum;
        row_out_v[i] = res;
    }
    for (int i = cols_v * 4 + tid; i < cols; i += BLOCK_THREADS) {
        output[row * cols + i] = __expf(input[row * cols + i] - row_max) * inv_sum;
    }
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    int size = input.size();
    if (size == 0) return {};
    int cols = size / row_count;
    size_t bytes = size * sizeof(float);
    std::vector<float> output(size);

    std::lock_guard<std::mutex> lock(mtx);
    if (!initialized) {
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        initialized = true;
    }

    if (bytes > cached_bytes) {
        if (d_in_cache) cudaFree(d_in_cache);
        if (d_out_cache) cudaFree(d_out_cache);
        cudaMalloc(&d_in_cache, bytes);
        cudaMalloc(&d_out_cache, bytes);
        cached_bytes = bytes;
    }

    cudaMemcpyAsync(d_in_cache, input.data(), bytes, cudaMemcpyHostToDevice, stream);

    softmax_final_kernel<<<row_count, BLOCK_THREADS, 0, stream>>>(d_in_cache, d_out_cache, cols);

    cudaMemcpyAsync(output.data(), d_out_cache, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    return output;
}