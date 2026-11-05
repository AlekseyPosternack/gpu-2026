#include "block_gemm_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int TILE_SIZE = 16;

void CheckCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) {
        CheckCuda(cudaMalloc(&ptr_, bytes), "cudaMalloc");
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

__global__ void BlockGemmKernel(
    const float* a,
    const float* b,
    float* c,
    int n) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    const int numTiles = (n + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < numTiles; ++tile) {
        const int aCol = tile * TILE_SIZE + threadIdx.x;
        const int bRow = tile * TILE_SIZE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] =
            (row < n && aCol < n) ? a[row * n + aCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] =
            (bRow < n && col < n) ? b[bRow * n + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

} // namespace

std::vector<float> BlockGemmCUDA(
    const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    if (n <= 0) {
        throw std::invalid_argument("Matrix size must be positive");
    }

    const std::size_t matrixSize =
        static_cast<std::size_t>(n) * static_cast<std::size_t>(n);

    if (a.size() != matrixSize || b.size() != matrixSize) {
        throw std::invalid_argument(
            "Input vectors must each contain n * n elements");
    }

    std::vector<float> c(matrixSize);
    const std::size_t bytes = matrixSize * sizeof(float);

    DeviceBuffer deviceA(bytes);
    DeviceBuffer deviceB(bytes);
    DeviceBuffer deviceC(bytes);

    CheckCuda(
        cudaMemcpy(deviceA.get(), a.data(), bytes, cudaMemcpyHostToDevice),
        "Copy matrix A to device");
    CheckCuda(
        cudaMemcpy(deviceB.get(), b.data(), bytes, cudaMemcpyHostToDevice),
        "Copy matrix B to device");

    const dim3 threads(TILE_SIZE, TILE_SIZE);
    const dim3 blocks(
        (n + TILE_SIZE - 1) / TILE_SIZE,
        (n + TILE_SIZE - 1) / TILE_SIZE);

    BlockGemmKernel<<<blocks, threads>>>(
        deviceA.get(), deviceB.get(), deviceC.get(), n);

    CheckCuda(cudaGetLastError(), "Launch BlockGemmKernel");
    CheckCuda(cudaDeviceSynchronize(), "Run BlockGemmKernel");

    CheckCuda(
        cudaMemcpy(c.data(), deviceC.get(), bytes, cudaMemcpyDeviceToHost),
        "Copy result to host");

    return c;
}
