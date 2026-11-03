#include "softmax_cuda.h"
#include <cuda_runtime.h>
#include <cfloat>
#include <cstddef>

namespace {

constexpr int BLOCK_THREADS = 256;

__global__ void SoftmaxComputeKernel(const float* in, float* out, int row_size) {
    __shared__ float s_data[BLOCK_THREADS];

    const float* current_in = in + static_cast<std::size_t>(blockIdx.x) * row_size;
    float* current_out = out + static_cast<std::size_t>(blockIdx.x) * row_size;

    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < row_size; i += BLOCK_THREADS) {
        max_val = fmaxf(max_val, current_in[i]);
    }

    s_data[threadIdx.x] = max_val;
    __syncthreads();
    for (int s = BLOCK_THREADS / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_data[threadIdx.x] = fmaxf(s_data[threadIdx.x], s_data[threadIdx.x + s]);
        }
        __syncthreads();
    }
    max_val = s_data[0];

    float sum_val = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += BLOCK_THREADS) {
        float exp_val = __expf(current_in[i] - max_val);
        current_out[i] = exp_val;
        sum_val += exp_val;
    }

    s_data[threadIdx.x] = sum_val;
    __syncthreads();
    for (int s = BLOCK_THREADS / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_data[threadIdx.x] += s_data[threadIdx.x + s];
        }
        __syncthreads();
    }
    sum_val = s_data[0];

    float inv_sum = 1.0f / sum_val;
    for (int i = threadIdx.x; i < row_size; i += BLOCK_THREADS) {
        current_out[i] *= inv_sum;
    }
}

struct GpuMemoryManager {
    float* d_input = nullptr;
    float* d_output = nullptr;
    cudaStream_t stream = nullptr;
    std::size_t size = 0;

    GpuMemoryManager() {
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    }

    ~GpuMemoryManager() {
        if (stream) cudaStreamSynchronize(stream);
        if (d_input) {
            cudaFree(d_input);
            cudaFree(d_output);
        }
        if (stream) cudaStreamDestroy(stream);
    }
};

GpuMemoryManager& GetManager() {
    thread_local GpuMemoryManager manager;
    return manager;
}

void EnsureMemory(GpuMemoryManager& mgr, std::size_t count) {
    if (mgr.size >= count) return;
    if (mgr.d_input != nullptr) {
        cudaFree(mgr.d_input);
        cudaFree(mgr.d_output);
    }
    std::size_t bytes = count * sizeof(float);
    cudaMalloc(&mgr.d_input, bytes);
    cudaMalloc(&mgr.d_output, bytes);
    mgr.size = count;
}

}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (input.empty() || row_count <= 0) {
        return {};
    }

    std::size_t total_elements = input.size();
    int row_size = static_cast<int>(total_elements / row_count);
    std::size_t bytes = total_elements * sizeof(float);

    GpuMemoryManager& mgr = GetManager();
    EnsureMemory(mgr, total_elements);

    cudaMemcpyAsync(mgr.d_input, input.data(), bytes, cudaMemcpyHostToDevice, mgr.stream);

    SoftmaxComputeKernel<<<row_count, BLOCK_THREADS, 0, mgr.stream>>>(mgr.d_input, mgr.d_output, row_size);

    std::vector<float> result(total_elements);
    cudaMemcpyAsync(result.data(), mgr.d_output, bytes, cudaMemcpyDeviceToHost, mgr.stream);
    cudaStreamSynchronize(mgr.stream);

    return result;
}