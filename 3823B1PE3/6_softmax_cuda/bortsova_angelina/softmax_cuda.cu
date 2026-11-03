#include "softmax_cuda.h"
#include <cuda_runtime.h>
#include <math_constants.h>

namespace {

constexpr int ROW_THREADS = 256;
constexpr int WARP_WIDTH = 32;
constexpr int WARPS_PER_ROW = ROW_THREADS / WARP_WIDTH;

__device__ float WarpMaximum(float value) {
    for (int offset = WARP_WIDTH / 2; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

__device__ float WarpSum(float value) {
    for (int offset = WARP_WIDTH / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__global__ void CollectRowStatistics(const float* values, float* row_maxima, float* inverse_sums, int row_size) {
    __shared__ float warp_maxima[WARPS_PER_ROW];
    __shared__ float warp_sums[WARPS_PER_ROW];
    int lane = threadIdx.x % WARP_WIDTH;
    int warp = threadIdx.x / WARP_WIDTH;
    size_t row_start = static_cast<size_t>(blockIdx.x) * row_size;
    float maximum = -CUDART_INF_F;

    for (int column = threadIdx.x; column < row_size; column += blockDim.x) {
        maximum = fmaxf(maximum, values[row_start + column]);
    }
    maximum = WarpMaximum(maximum);
    if (lane == 0) {
        warp_maxima[warp] = maximum;
    }
    __syncthreads();

    if (warp == 0) {
        maximum = lane < WARPS_PER_ROW ? warp_maxima[lane] : -CUDART_INF_F;
        maximum = WarpMaximum(maximum);
        if (lane == 0) {
            row_maxima[blockIdx.x] = maximum;
        }
    }
    __syncthreads();

    maximum = row_maxima[blockIdx.x];
    float sum = 0.0f;
    for (int column = threadIdx.x; column < row_size; column += blockDim.x) {
        sum += __expf(values[row_start + column] - maximum);
    }
    sum = WarpSum(sum);
    if (lane == 0) {
        warp_sums[warp] = sum;
    }
    __syncthreads();

    if (warp == 0) {
        sum = lane < WARPS_PER_ROW ? warp_sums[lane] : 0.0f;
        sum = WarpSum(sum);
        if (lane == 0) {
            inverse_sums[blockIdx.x] = 1.0f / sum;
        }
    }
}

__global__ void NormalizeElements(float* values, const float* row_maxima, const float* inverse_sums, int row_size, size_t element_count) {
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < element_count) {
        size_t row = index / row_size;
        values[index] = __expf(values[index] - row_maxima[row]) * inverse_sums[row];
    }
}

struct SoftmaxWorkspace {
    float* values = nullptr;
    float* statistics = nullptr;
    size_t value_capacity = 0;
    int row_capacity = 0;
    cudaStream_t stream;

    SoftmaxWorkspace() {
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    }

    ~SoftmaxWorkspace() {
        cudaFree(values);
        cudaFree(statistics);
        cudaStreamDestroy(stream);
    }

    void Reserve(size_t element_count, int row_count) {
        if (element_count > value_capacity) {
            cudaFree(values);
            cudaMalloc(&values, element_count * sizeof(float));
            value_capacity = element_count;
        }
        if (row_count > row_capacity) {
            cudaFree(statistics);
            cudaMalloc(&statistics, 2 * static_cast<size_t>(row_count) * sizeof(float));
            row_capacity = row_count;
        }
    }
};

}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (input.empty()) {
        return {};
    }

    static SoftmaxWorkspace workspace;
    workspace.Reserve(input.size(), row_count);
    size_t byte_count = input.size() * sizeof(float);
    int row_size = static_cast<int>(input.size() / row_count);
    float* row_maxima = workspace.statistics;
    float* inverse_sums = row_maxima + row_count;
    cudaMemcpyAsync(workspace.values, input.data(), byte_count, cudaMemcpyHostToDevice, workspace.stream);

    CollectRowStatistics<<<row_count, ROW_THREADS, 0, workspace.stream>>>(workspace.values, row_maxima, inverse_sums, row_size);
    unsigned int block_count = static_cast<unsigned int>((input.size() + ROW_THREADS - 1) / ROW_THREADS);
    NormalizeElements<<<block_count, ROW_THREADS, 0, workspace.stream>>>(workspace.values, row_maxima, inverse_sums, row_size, input.size());

    std::vector<float> result(input.size());
    cudaMemcpyAsync(result.data(), workspace.values, byte_count, cudaMemcpyDeviceToHost, workspace.stream);
    cudaStreamSynchronize(workspace.stream);
    return result;
}
