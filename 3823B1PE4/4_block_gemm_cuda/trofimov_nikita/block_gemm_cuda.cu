#include "block_gemm_cuda.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace {

constexpr int TILE = 16;

__global__ void BlockGemmKernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    for (int tile = 0; tile < n; tile += TILE) {

        As[ty][tx] = (row < n && tile + tx < n)
            ? A[static_cast<std::size_t>(row) * n + tile + tx]
            : 0.0f;

        Bs[ty][tx] = (tile + ty < n && col < n)
            ? B[static_cast<std::size_t>(tile + ty) * n + col]
            : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        C[static_cast<std::size_t>(row) * n + col] = sum;
    }
}

void CheckCUDA(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(error)
        );
    }
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) {
        CheckCUDA(
            cudaMalloc(reinterpret_cast<void**>(&ptr), bytes),
            "cudaMalloc"
        );
    }

    ~DeviceBuffer() {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    float* get() const {
        return ptr;
    }

    void release() {
        if (ptr != nullptr) {
            float* p = ptr;
            ptr = nullptr;
            CheckCUDA(cudaFree(p), "cudaFree");
        }
    }

private:
    float* ptr = nullptr;
};

} // namespace

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n < 0) {
        throw std::invalid_argument("Matrix size cannot be negative");
    }

    const std::size_t count =
        static_cast<std::size_t>(n) * static_cast<std::size_t>(n);

    if (a.size() != count || b.size() != count) {
        throw std::invalid_argument("Matrices must have exactly n*n elements");
    }

    std::vector<float> c(count, 0.0f);

    if (n == 0) {
        return c;
    }

    const std::size_t bytes = count * sizeof(float);

    DeviceBuffer deviceMemory(3 * bytes);

    float* d_a = deviceMemory.get();
    float* d_b = d_a + count;
    float* d_c = d_b + count;

    CheckCUDA(
        cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice),
        "cudaMemcpy A"
    );

    CheckCUDA(
        cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice),
        "cudaMemcpy B"
    );

    const dim3 block(TILE, TILE);

    const dim3 grid(
        (n + TILE - 1) / TILE,
        (n + TILE - 1) / TILE
    );

    BlockGemmKernel<<<grid, block>>>(d_a, d_b, d_c, n);

    CheckCUDA(cudaGetLastError(), "BlockGemmKernel launch");

    CheckCUDA(
        cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost),
        "cudaMemcpy C"
    );

    deviceMemory.release();

    return c;
}