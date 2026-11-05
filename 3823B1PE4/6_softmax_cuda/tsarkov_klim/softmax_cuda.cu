#include "softmax_cuda.h"

#include <cuda_runtime.h>

#include <climits>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int THREADS_PER_BLOCK = 256;

void CheckCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) {
        CheckCuda(
            cudaMalloc(reinterpret_cast<void**>(&ptr_), bytes),
            "cudaMalloc");
    }

    ~DeviceBuffer() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    float* get() const {
        return ptr_;
    }

private:
    float* ptr_ = nullptr;
};

__global__ void SoftmaxKernel(
    const float* input,
    float* output,
    int row_size) {
    __shared__ float reduction[THREADS_PER_BLOCK];

    const int row = blockIdx.x;
    const int thread = threadIdx.x;
    const int rowStart = row * row_size;

    float localMax = -3.402823466e+38F;
    for (int col = thread; col < row_size; col += blockDim.x) {
        localMax = fmaxf(localMax, input[rowStart + col]);
    }

    reduction[thread] = localMax;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread < stride) {
            reduction[thread] =
                fmaxf(reduction[thread], reduction[thread + stride]);
        }
        __syncthreads();
    }

    const float rowMax = reduction[0];

    float localSum = 0.0f;
    for (int col = thread; col < row_size; col += blockDim.x) {
        localSum += expf(input[rowStart + col] - rowMax);
    }

    reduction[thread] = localSum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread < stride) {
            reduction[thread] += reduction[thread + stride];
        }
        __syncthreads();
    }

    const float rowSum = reduction[0];

    for (int col = thread; col < row_size; col += blockDim.x) {
        output[rowStart + col] =
            expf(input[rowStart + col] - rowMax) / rowSum;
    }
}

} // namespace

std::vector<float> SoftmaxCUDA(
    const std::vector<float>& input,
    int row_count) {
    if (row_count <= 0) {
        throw std::invalid_argument("row_count must be positive");
    }

    if (input.empty() ||
        input.size() % static_cast<std::size_t>(row_count) != 0) {
        throw std::invalid_argument(
            "Input must be non-empty and divisible by row_count");
    }

    const std::size_t rowSize = input.size() / row_count;
    if (rowSize > static_cast<std::size_t>(INT_MAX)) {
        throw std::invalid_argument("A row is too large");
    }

    std::vector<float> output(input.size());
    const std::size_t bytes = input.size() * sizeof(float);

    DeviceBuffer deviceInput(bytes);
    DeviceBuffer deviceOutput(bytes);

    CheckCuda(
        cudaMemcpy(
            deviceInput.get(),
            input.data(),
            bytes,
            cudaMemcpyHostToDevice),
        "Copy input to device");

    SoftmaxKernel<<<row_count, THREADS_PER_BLOCK>>>(
        deviceInput.get(),
        deviceOutput.get(),
        static_cast<int>(rowSize));

    CheckCuda(cudaGetLastError(), "Launch SoftmaxKernel");
    CheckCuda(cudaDeviceSynchronize(), "Run SoftmaxKernel");

    CheckCuda(
        cudaMemcpy(
            output.data(),
            deviceOutput.get(),
            bytes,
            cudaMemcpyDeviceToHost),
        "Copy output to host");

    return output;
}
