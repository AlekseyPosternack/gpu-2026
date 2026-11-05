#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace {

constexpr int kBlockSize = 32;

__global__ void NaiveGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n || col >= n) {
        return;
    }

    float sum = 0.0f;
    const int rowOffset = row * n;

    // Threads in a warp access neighboring columns of B, making each
    // iteration's B load coalesced. A is broadcast from the same row.
    #pragma unroll 4
    for (int k = 0; k < n; ++k) {
        sum += a[rowOffset + k] * b[k * n + col];
    }

    c[rowOffset + col] = sum;
}

void CheckCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

} // namespace

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n < 0) {
        throw std::invalid_argument("Matrix size n must be non-negative");
    }
    const std::size_t dimension = static_cast<std::size_t>(n);
    if (dimension != 0 &&
        dimension > std::numeric_limits<std::size_t>::max() / dimension) {
        throw std::invalid_argument("Matrix size is too large");
    }
    const std::size_t elements = dimension * dimension;

    if (a.size() != elements || b.size() != elements) {
        throw std::invalid_argument("Input matrices must have size n*n");
    }

    std::vector<float> result(elements);
    if (elements == 0) {
        return result;
    }

    float* deviceA = nullptr;
    float* deviceB = nullptr;
    float* deviceC = nullptr;
    const std::size_t bytes = elements * sizeof(float);

    try {
        CheckCuda(cudaMalloc(&deviceA, bytes), "cudaMalloc(A)");
        CheckCuda(cudaMalloc(&deviceB, bytes), "cudaMalloc(B)");
        CheckCuda(cudaMalloc(&deviceC, bytes), "cudaMalloc(C)");

        CheckCuda(cudaMemcpy(deviceA, a.data(), bytes, cudaMemcpyHostToDevice),
                  "cudaMemcpy(A)");
        CheckCuda(cudaMemcpy(deviceB, b.data(), bytes, cudaMemcpyHostToDevice),
                  "cudaMemcpy(B)");

        const dim3 block(kBlockSize, kBlockSize);
        const dim3 grid((n + kBlockSize - 1) / kBlockSize,
                        (n + kBlockSize - 1) / kBlockSize);
        NaiveGemmKernel<<<grid, block>>>(deviceA, deviceB, deviceC, n);
        CheckCuda(cudaGetLastError(), "NaiveGemmKernel launch");

        CheckCuda(cudaMemcpy(result.data(), deviceC, bytes,
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(C)");

        CheckCuda(cudaFree(deviceC), "cudaFree(C)");
        deviceC = nullptr;
        CheckCuda(cudaFree(deviceB), "cudaFree(B)");
        deviceB = nullptr;
        CheckCuda(cudaFree(deviceA), "cudaFree(A)");
        deviceA = nullptr;
    } catch (...) {
        if (deviceC != nullptr) {
            cudaFree(deviceC);
        }
        if (deviceB != nullptr) {
            cudaFree(deviceB);
        }
        if (deviceA != nullptr) {
            cudaFree(deviceA);
        }
        throw;
    }

    return result;
}