#include "softmax_cuda.h"

#include <cuda_runtime.h>
#include <float.h>
#include <stdexcept>
#include <string>

namespace {

constexpr int kThreadsPerBlock = 256;

__global__ void RowMaxKernel(const float* input,
                             float* row_max,
                             int row_count,
                             int row_size) {
    const int row = blockIdx.x;
    if (row >= row_count) {
        return;
    }

    const int row_offset = row * row_size;
    extern __shared__ float shared_values[];
    float local_max = -FLT_MAX;

    for (int idx = threadIdx.x; idx < row_size; idx += blockDim.x) {
        const float value = input[row_offset + idx];
        if (value > local_max) {
            local_max = value;
        }
    }

    shared_values[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float lhs = shared_values[threadIdx.x];
            const float rhs = shared_values[threadIdx.x + stride];
            shared_values[threadIdx.x] = lhs > rhs ? lhs : rhs;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        row_max[row] = shared_values[0];
    }
}

__global__ void SoftmaxExpAndSumKernel(const float* input,
                                      const float* row_max,
                                      float* exp_values,
                                      float* row_sum,
                                      int row_count,
                                      int row_size) {
    const int row = blockIdx.x;
    if (row >= row_count) {
        return;
    }

    const int row_offset = row * row_size;
    extern __shared__ float shared_values[];

    float local_sum = 0.0f;
    for (int idx = threadIdx.x; idx < row_size; idx += blockDim.x) {
        const float value = expf(input[row_offset + idx] - row_max[row]);
        exp_values[row_offset + idx] = value;
        local_sum += value;
    }

    shared_values[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_values[threadIdx.x] += shared_values[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        row_sum[row] = shared_values[0];
    }
}

__global__ void SoftmaxFinalizeKernel(const float* exp_values,
                                    const float* row_sum,
                                    float* output,
                                    int row_count,
                                    int row_size) {
    const int row = blockIdx.x;
    if (row >= row_count) {
        return;
    }

    const int row_offset = row * row_size;
    for (int idx = threadIdx.x; idx < row_size; idx += blockDim.x) {
        output[row_offset + idx] = exp_values[row_offset + idx] / row_sum[row];
    }
}

}  // namespace

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (row_count <= 0) {
        return {};
    }
    if (input.empty()) {
        return {};
    }
    if (static_cast<int>(input.size()) % row_count != 0) {
        throw std::invalid_argument("input size must be divisible by row_count");
    }

    const int row_size = static_cast<int>(input.size()) / row_count;

    std::vector<float> output(input.size());
    float* d_input = nullptr;
    float* d_row_max = nullptr;
    float* d_exp = nullptr;
    float* d_row_sum = nullptr;
    float* d_output = nullptr;

    const size_t input_size_bytes = input.size() * sizeof(float);
    const size_t row_max_size_bytes = static_cast<size_t>(row_count) * sizeof(float);

    cudaError_t cuda_err = cudaMalloc(&d_input, input_size_bytes);
    if (cuda_err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc input failed: ") + cudaGetErrorString(cuda_err));
    }

    cuda_err = cudaMalloc(&d_row_max, row_max_size_bytes);
    if (cuda_err != cudaSuccess) {
        cudaFree(d_input);
        throw std::runtime_error(std::string("cudaMalloc row_max failed: ") + cudaGetErrorString(cuda_err));
    }

    cuda_err = cudaMalloc(&d_exp, input_size_bytes);
    if (cuda_err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_row_max);
        throw std::runtime_error(std::string("cudaMalloc exp buffer failed: ") + cudaGetErrorString(cuda_err));
    }

    cuda_err = cudaMalloc(&d_row_sum, row_max_size_bytes);
    if (cuda_err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_row_max);
        cudaFree(d_exp);
        throw std::runtime_error(std::string("cudaMalloc row_sum failed: ") + cudaGetErrorString(cuda_err));
    }

    cuda_err = cudaMalloc(&d_output, input_size_bytes);
    if (cuda_err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_row_max);
        cudaFree(d_exp);
        cudaFree(d_row_sum);
        throw std::runtime_error(std::string("cudaMalloc output failed: ") + cudaGetErrorString(cuda_err));
    }

    cuda_err = cudaMemcpy(d_input, input.data(), input_size_bytes, cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_row_max);
        cudaFree(d_exp);
        cudaFree(d_row_sum);
        cudaFree(d_output);
        throw std::runtime_error(std::string("cudaMemcpy input failed: ") + cudaGetErrorString(cuda_err));
    }

    RowMaxKernel<<<row_count, kThreadsPerBlock, kThreadsPerBlock * sizeof(float)>>>(d_input, d_row_max, row_count, row_size);
    cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_row_max);
        cudaFree(d_exp);
        cudaFree(d_row_sum);
        cudaFree(d_output);
        throw std::runtime_error(std::string("RowMaxKernel launch failed: ") + cudaGetErrorString(cuda_err));
    }

    SoftmaxExpAndSumKernel<<<row_count, kThreadsPerBlock, kThreadsPerBlock * sizeof(float)>>>(
        d_input, d_row_max, d_exp, d_row_sum, row_count, row_size);
    cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_row_max);
        cudaFree(d_exp);
        cudaFree(d_row_sum);
        cudaFree(d_output);
        throw std::runtime_error(std::string("SoftmaxExpAndSumKernel launch failed: ") + cudaGetErrorString(cuda_err));
    }

    SoftmaxFinalizeKernel<<<row_count, kThreadsPerBlock>>>(d_exp, d_row_sum, d_output, row_count, row_size);
    cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_row_max);
        cudaFree(d_exp);
        cudaFree(d_row_sum);
        cudaFree(d_output);
        throw std::runtime_error(std::string("SoftmaxFinalizeKernel launch failed: ") + cudaGetErrorString(cuda_err));
    }

    cuda_err = cudaMemcpy(output.data(), d_output, input_size_bytes, cudaMemcpyDeviceToHost);
    if (cuda_err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_row_max);
        cudaFree(d_exp);
        cudaFree(d_row_sum);
        cudaFree(d_output);
        throw std::runtime_error(std::string("cudaMemcpy output failed: ") + cudaGetErrorString(cuda_err));
    }

    cudaFree(d_input);
    cudaFree(d_row_max);
    cudaFree(d_exp);
    cudaFree(d_row_sum);
    cudaFree(d_output);

    return output;
}
