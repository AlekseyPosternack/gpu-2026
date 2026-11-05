#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace {

constexpr int BLOCK_X = 32;
constexpr int BLOCK_Y = 8;
constexpr int ROWS_PER_THREAD = 4;

// Each thread calculates several elements of matrix C.
__global__ void NaiveGemmKernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int n
) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row =
        (blockIdx.y * blockDim.y + threadIdx.y) * ROWS_PER_THREAD;

    if (col >= n || row >= n) {
        return;
    }

    float acc[ROWS_PER_THREAD] = {};

    // Vectorized calculation for matrices with n >= 4.
    if (n >= 4) {
        for (int k = 0; k < n; k += 4) {

            const float b0 = B[static_cast<std::size_t>(k) * n + col];
            const float b1 = B[static_cast<std::size_t>(k + 1) * n + col];
            const float b2 = B[static_cast<std::size_t>(k + 2) * n + col];
            const float b3 = B[static_cast<std::size_t>(k + 3) * n + col];

            #pragma unroll
            for (int r = 0; r < ROWS_PER_THREAD; ++r) {
                if (row + r < n) {
                    const float4 av =
                        *reinterpret_cast<const float4*>(
                            A + static_cast<std::size_t>(row + r) * n + k
                        );

                    acc[r] += av.x * b0
                            + av.y * b1
                            + av.z * b2
                            + av.w * b3;
                }
            }
        }
    }
    else {
        // Scalar calculation for small matrices.
        for (int k = 0; k < n; ++k) {
            const float bk =
                B[static_cast<std::size_t>(k) * n + col];

            #pragma unroll
            for (int r = 0; r < ROWS_PER_THREAD; ++r) {
                if (row + r < n) {
                    acc[r] +=
                        A[static_cast<std::size_t>(row + r) * n + k] * bk;
                }
            }
        }
    }

    // Store calculated elements.
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        if (row + r < n) {
            C[static_cast<std::size_t>(row + r) * n + col] = acc[r];
        }
    }
}

void CheckCUDA(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(error)
        );
    }
}

} // namespace

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) {
        return {};
    }

    const std::size_t count = static_cast<std::size_t>(n) * n;

    if (a.size() != count || b.size() != count) {
        throw std::invalid_argument("Invalid matrix size");
    }

    std::vector<float> c(count, 0.0f);

    const std::size_t bytes = count * sizeof(float);

    float* device_memory = nullptr;

    try {
        // Allocate memory for all three matrices.
        CheckCUDA(
            cudaMalloc(
                reinterpret_cast<void**>(&device_memory),
                3 * bytes
            ),
            "cudaMalloc"
        );

        float* d_a = device_memory;
        float* d_b = device_memory + count;
        float* d_c = device_memory + 2 * count;

        // Copy input matrices to GPU.
        CheckCUDA(
            cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy A"
        );

        CheckCUDA(
            cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy B"
        );

        // Configure CUDA grid and blocks.
        const dim3 block(BLOCK_X, BLOCK_Y);

        const int rows_per_block = BLOCK_Y * ROWS_PER_THREAD;

        const dim3 grid(
            (n + BLOCK_X - 1) / BLOCK_X,
            (n + rows_per_block - 1) / rows_per_block
        );

        // Launch matrix multiplication kernel.
        NaiveGemmKernel<<<grid, block>>>(d_a, d_b, d_c, n);

        CheckCUDA(cudaGetLastError(), "Kernel launch");

        // Copy result matrix back to CPU.
        CheckCUDA(
            cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy C"
        );

        // Release GPU memory.
        cudaError_t free_error = cudaFree(device_memory);
        device_memory = nullptr;

        CheckCUDA(free_error, "cudaFree");
    }
    catch (...) {
        if (device_memory != nullptr) {
            cudaFree(device_memory);
        }
        throw;
    }

    return c;
}