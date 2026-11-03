#include "softmax_cuda.h"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void CheckCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

class DeviceStorage {
public:
    explicit DeviceStorage(std::size_t bytes) {
        void* memory = nullptr;
        CheckCuda(cudaMalloc(&memory, bytes), "cudaMalloc");
        data_ = static_cast<float*>(memory);
    }

    ~DeviceStorage() {
        cudaFree(data_);
    }

    DeviceStorage(const DeviceStorage&) = delete;
    DeviceStorage& operator=(const DeviceStorage&) = delete;

    float* data() const { return data_; }

private:
    float* data_ = nullptr;
};

__global__ void SoftmaxRowsKernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  std::size_t row_size) {
    extern __shared__ float scratch[];
    const unsigned int thread = threadIdx.x;
    const std::size_t row_offset = static_cast<std::size_t>(blockIdx.x) * row_size;

    float local_max = -CUDART_INF_F;
    for (std::size_t col = thread; col < row_size; col += blockDim.x) {
        local_max = fmaxf(local_max, input[row_offset + col]);
    }
    scratch[thread] = local_max;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread < stride) {
            scratch[thread] = fmaxf(scratch[thread], scratch[thread + stride]);
        }
        __syncthreads();
    }
    const float row_max = scratch[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (std::size_t col = thread; col < row_size; col += blockDim.x) {
        const float value = expf(input[row_offset + col] - row_max);
        output[row_offset + col] = value;
        local_sum += value;
    }
    scratch[thread] = local_sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread < stride) {
            scratch[thread] += scratch[thread + stride];
        }
        __syncthreads();
    }
    const float inverse_sum = 1.0f / scratch[0];

    for (std::size_t col = thread; col < row_size; col += blockDim.x) {
        output[row_offset + col] *= inverse_sum;
    }
}

} // namespace

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (row_count < 0) {
        throw std::invalid_argument("row_count must be nonnegative");
    }
    if (row_count == 0) {
        if (!input.empty()) {
            throw std::invalid_argument("nonempty input requires at least one row");
        }
        return {};
    }

    const std::size_t rows = static_cast<std::size_t>(row_count);
    if (input.size() % rows != 0) {
        throw std::invalid_argument("input size must be divisible by row_count");
    }
    if (input.empty()) {
        return {};
    }
    if (input.size() > std::numeric_limits<std::size_t>::max() / (2 * sizeof(float))) {
        throw std::length_error("input size exceeds addressable device memory");
    }

    const std::size_t row_size = input.size() / rows;
    const std::size_t bytes = input.size() * sizeof(float);
    DeviceStorage storage(2 * bytes);
    float* const device_input = storage.data();
    float* const device_output = device_input + input.size();
    CheckCuda(cudaMemcpy(device_input, input.data(), bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy input to device");

    unsigned int threads = 1;
    while (threads < row_size && threads < 256) {
        threads <<= 1;
    }
    SoftmaxRowsKernel<<<row_count, threads, threads * sizeof(float)>>>(
        device_input, device_output, row_size);
    CheckCuda(cudaGetLastError(), "SoftmaxRowsKernel launch");

    std::vector<float> result(input.size());
    CheckCuda(cudaMemcpy(result.data(), device_output, bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy output to host");
    return result;
}
