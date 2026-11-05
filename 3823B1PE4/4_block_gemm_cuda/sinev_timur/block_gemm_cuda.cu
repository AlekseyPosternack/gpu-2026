#include "block_gemm_cuda.h"

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace {

// 32x32 gives coalesced global loads and uses 8 KiB of shared memory for
// each input tile (well within the shared-memory budget of modern GPUs).
constexpr int kTileSize = 32;

__global__ void BlockGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                int n) {
    __shared__ float tileA[kTileSize][kTileSize];
    __shared__ float tileB[kTileSize][kTileSize];

    const int localRow = threadIdx.y;
    const int localCol = threadIdx.x;
    const int row = blockIdx.y * kTileSize + localRow;
    const int col = blockIdx.x * kTileSize + localCol;

    float sum = 0.0f;

    // For each K tile, neighboring threads read neighboring global elements.
    for (int tile = 0; tile < n; tile += kTileSize) {
        tileA[localRow][localCol] =
            (row < n && tile + localCol < n)
                ? a[row * n + tile + localCol]
                : 0.0f;
        tileB[localRow][localCol] =
            (tile + localRow < n && col < n)
                ? b[(tile + localRow) * n + col]
                : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < kTileSize; ++k) {
            sum += tileA[localRow][k] * tileB[k][localCol];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

void CheckCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

} // namespace

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
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

        const dim3 block(kTileSize, kTileSize);
        const dim3 grid((n + kTileSize - 1) / kTileSize,
                        (n + kTileSize - 1) / kTileSize);
        BlockGemmKernel<<<grid, block>>>(deviceA, deviceB, deviceC, n);
        CheckCuda(cudaGetLastError(), "BlockGemmKernel launch");

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
