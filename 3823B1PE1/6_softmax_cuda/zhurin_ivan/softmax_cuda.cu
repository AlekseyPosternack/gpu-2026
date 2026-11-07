#include "softmax_cuda.h"

#include <cfloat>
#include <cuda_runtime.h>

namespace {

constexpr int kBlockSize = 256;

__global__ void SoftmaxKernel(float* data, int row_size) {
    extern __shared__ float sdata[];

    const int row = blockIdx.x;
    float* row_data = data + static_cast<size_t>(row) * row_size;

    float local_max = -FLT_MAX;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        local_max = fmaxf(local_max, row_data[i]);
    }
    sdata[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float row_max = sdata[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        float e = expf(row_data[i] - row_max);
        row_data[i] = e;
        local_sum += e;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float row_sum = sdata[0];
    __syncthreads();

    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        row_data[i] /= row_sum;
    }
}

}  // namespace

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    const size_t size = input.size();
    if (size == 0 || row_count <= 0) {
        return {};
    }
    const int row_size = static_cast<int>(size / static_cast<size_t>(row_count));
    const size_t bytes = size * sizeof(float);

    float* d_data = nullptr;
    cudaMalloc(&d_data, bytes);
    cudaMemcpy(d_data, input.data(), bytes, cudaMemcpyHostToDevice);

    const int block = kBlockSize;
    const size_t shared_bytes = static_cast<size_t>(block) * sizeof(float);
    SoftmaxKernel<<<row_count, block, shared_bytes>>>(d_data, row_size);

    std::vector<float> output(size);
    cudaMemcpy(output.data(), d_data, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_data);

    return output;
}
