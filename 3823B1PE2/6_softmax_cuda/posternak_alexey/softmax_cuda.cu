#include "softmax_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_THREADS = 256;
constexpr int MAX_WARPS_PER_BLOCK = BLOCK_THREADS / WARP_SIZE;

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

template <typename ReduceOp>
__device__ __forceinline__ float block_reduce(float val,
                                              float* sdata,
                                              int tid,
                                              ReduceOp op,
                                              float identity) {
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    val = op(val);

    if (lane_id == 0)
        sdata[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane_id < MAX_WARPS_PER_BLOCK) ? sdata[lane_id] : identity;
        v = op(v);
        if (lane_id == 0) sdata[0] = v;
    }
    __syncthreads();

    return sdata[0];
}

__device__ __forceinline__ float block_reduce_max(float val, float* sdata, int tid) {
    return block_reduce(val, sdata, tid, warp_reduce_max, -INFINITY);
}

__device__ __forceinline__ float block_reduce_sum(float val, float* sdata, int tid) {
    return block_reduce(val, sdata, tid, warp_reduce_sum, 0.0f);
}

__global__ void softmax_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int row_size) {
    extern __shared__ float shared[];

    const int row        = blockIdx.x;
    const int tid        = threadIdx.x;
    const int row_offset = row * row_size;

    float local_max = -INFINITY;
    for (int i = tid; i < row_size; i += BLOCK_THREADS)
        local_max = fmaxf(local_max, input[row_offset + i]);

    const float row_max = block_reduce_max(local_max, shared, tid);

    float local_sum = 0.0f;
    for (int i = tid; i < row_size; i += BLOCK_THREADS)
        local_sum += expf(input[row_offset + i] - row_max);

    const float row_sum = block_reduce_sum(local_sum, shared, tid);

    const float inv_sum = 1.0f / row_sum;
    for (int i = tid; i < row_size; i += BLOCK_THREADS) {
        float exp_val = expf(input[row_offset + i] - row_max);
        output[row_offset + i] = exp_val * inv_sum;
    }
}

static float* d_input = nullptr;
static float* d_output = nullptr;
static size_t allocated_size = 0;

static void cleanup_cuda_resources() {
    if (d_input)
        cudaFree(d_input);
    if (d_output)
        cudaFree(d_output);
    d_input = d_output = nullptr;
    allocated_size = 0;
}

static void ensure_device_memory(size_t required_size) {
    if (required_size <= allocated_size)
        return;

    if (d_input)
        cudaFree(d_input);
    if (d_output)
        cudaFree(d_output);

    cudaMalloc(&d_input,  required_size * sizeof(float));
    cudaMalloc(&d_output, required_size * sizeof(float));
    allocated_size = required_size;

    static bool cleanup_registered = false;
    if (!cleanup_registered) {
        std::atexit(cleanup_cuda_resources);
        cleanup_registered = true;
    }
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (input.empty() || row_count == 0)
        return {};

    const size_t total_size = input.size();
    const int    row_size   = static_cast<int>(total_size / row_count);

    ensure_device_memory(total_size);

    cudaMemcpyAsync(d_input, input.data(),
                    total_size * sizeof(float),
                    cudaMemcpyHostToDevice, 0);

    const int shared_mem_bytes = MAX_WARPS_PER_BLOCK * sizeof(float);
    dim3 grid(row_count);
    dim3 block(BLOCK_THREADS);

    softmax_kernel<<<grid, block, shared_mem_bytes>>>(d_input, d_output, row_size);

    std::vector<float> output(total_size);

    cudaMemcpyAsync(output.data(), d_output,
                    total_size * sizeof(float),
                    cudaMemcpyDeviceToHost, 0);

    cudaDeviceSynchronize();
    return output;
}
