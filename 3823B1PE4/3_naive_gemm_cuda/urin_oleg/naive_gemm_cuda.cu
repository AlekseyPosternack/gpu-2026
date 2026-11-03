#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace {

void CheckCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) {
        void* memory = nullptr;
        CheckCuda(cudaMalloc(&memory, bytes), "cudaMalloc");
        data_ = static_cast<float*>(memory);
    }

    ~DeviceBuffer() {
        cudaFree(data_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    float* data() const { return data_; }

private:
    float* data_ = nullptr;
};

__global__ void NaiveGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                int n) {
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= static_cast<unsigned int>(n) || col >= static_cast<unsigned int>(n)) {
        return;
    }

    const std::size_t row_offset = static_cast<std::size_t>(row) * n;
    float sum = 0.0f;
#pragma unroll 4
    for (int k = 0; k < n; ++k) {
        sum = fmaf(a[row_offset + k], b[static_cast<std::size_t>(k) * n + col], sum);
    }
    c[row_offset + col] = sum;
}

} // namespace

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n < 0) {
        throw std::invalid_argument("n must be nonnegative");
    }

    const std::size_t count = static_cast<std::size_t>(n) * n;
    if (a.size() != count || b.size() != count) {
        throw std::invalid_argument("both matrices must contain n * n elements");
    }
    if (count == 0) {
        return {};
    }

    const std::size_t bytes = count * sizeof(float);
    DeviceBuffer device_a(bytes);
    DeviceBuffer device_b(bytes);
    DeviceBuffer device_c(bytes);

    CheckCuda(cudaMemcpy(device_a.data(), a.data(), bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy A to device");
    CheckCuda(cudaMemcpy(device_b.data(), b.data(), bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy B to device");

    const unsigned int width = std::min(static_cast<unsigned int>(n), 32u);
    const unsigned int height = std::min(static_cast<unsigned int>(n), 256u / width);
    const dim3 threads(width, height);
    const dim3 blocks((static_cast<unsigned int>(n) + width - 1) / width,
                      (static_cast<unsigned int>(n) + height - 1) / height);
    NaiveGemmKernel<<<blocks, threads>>>(device_a.data(), device_b.data(), device_c.data(), n);
    CheckCuda(cudaGetLastError(), "NaiveGemmKernel launch");

    std::vector<float> result(count);
    CheckCuda(cudaMemcpy(result.data(), device_c.data(), bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy C to host");
    return result;
}
