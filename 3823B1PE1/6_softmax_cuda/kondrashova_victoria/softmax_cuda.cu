#include "softmax_cuda.h"

#include <cuda_runtime.h>
#include <cfloat>

namespace {

constexpr int kBlockSize = 256;

__global__ void SoftmaxKernel(const float* __restrict__ input,
                              float* __restrict__ output,
                              int row_size) {
    __shared__ float shared[kBlockSize];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const float* row_input = input + row * row_size;
    float* row_output = output + row * row_size;

    float local_max = -FLT_MAX;

    for (int i = tid; i < row_size; i += kBlockSize) {
        local_max = fmaxf(local_max, row_input[i]);
    }

    shared[tid] = local_max;
    __syncthreads();

    for (int stride = kBlockSize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }

    const float row_max = shared[0];
    __syncthreads();

    float local_sum = 0.0f;

    for (int i = tid; i < row_size; i += kBlockSize) {
        const float value = __expf(row_input[i] - row_max);
        row_output[i] = value;
        local_sum += value;
    }

    shared[tid] = local_sum;
    __syncthreads();

    for (int stride = kBlockSize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    const float row_sum = shared[0];

    for (int i = tid; i < row_size; i += kBlockSize) {
        row_output[i] /= row_sum;
    }
}

} // namespace

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    const int row_size = static_cast<int>(input.size()) / row_count;

    std::vector<float> output(input.size());

    if (input.empty()) {
        return output;
    }

    float* d_input = nullptr;
    float* d_output = nullptr;

    const size_t bytes = input.size() * sizeof(float);

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);

    SoftmaxKernel<<<row_count, kBlockSize>>>(d_input, d_output, row_size);

    cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}
