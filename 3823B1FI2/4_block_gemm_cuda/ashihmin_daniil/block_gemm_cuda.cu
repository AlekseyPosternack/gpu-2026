#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <cstddef>

namespace {

template <int TILE_DIM>
__global__ void BlockGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                int n) {
    __shared__ float s_a[TILE_DIM][TILE_DIM];
    __shared__ float s_b[TILE_DIM][TILE_DIM];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = blockIdx.y * TILE_DIM + ty;
    const int col = blockIdx.x * TILE_DIM + tx;

    float sum = 0.0f;
    const int num_tiles = (n + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < num_tiles; ++t) {
        const int a_col = t * TILE_DIM + tx;
        const int b_row = t * TILE_DIM + ty;

        s_a[ty][tx] = (row < n && a_col < n) ? a[row * n + a_col] : 0.0f;
        s_b[ty][tx] = (b_row < n && col < n) ? b[b_row * n + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            sum = fmaf(s_a[ty][k], s_b[k][tx], sum);
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

struct PersistentGpuMemory {
    float* raw_data = nullptr;
    size_t capacity_bytes = 0;

    float* Ensure(size_t required_bytes) {
        if (required_bytes > capacity_bytes) {
            if (raw_data != nullptr) {
                cudaFree(raw_data);
            }
            cudaMalloc(&raw_data, required_bytes);
            capacity_bytes = required_bytes;
        }
        return raw_data;
    }

    ~PersistentGpuMemory() {
        if (raw_data != nullptr) {
            cudaFree(raw_data);
        }
    }
};

PersistentGpuMemory g_memory_pool;

} // namespace

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) {
        return {};
    }

    const size_t total_elements = static_cast<size_t>(n) * n;
    const size_t matrix_bytes = total_elements * sizeof(float);

    float* d_all = g_memory_pool.Ensure(3 * matrix_bytes);
    float* d_a = d_all;
    float* d_b = d_all + total_elements;
    float* d_c = d_all + 2 * total_elements;

    cudaMemcpyAsync(d_a, a.data(), matrix_bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_b, b.data(), matrix_bytes, cudaMemcpyHostToDevice);

    if (n <= 16) {
        constexpr int TILE = 16;
        dim3 block(TILE, TILE);
        dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
        BlockGemmKernel<TILE><<<grid, block>>>(d_a, d_b, d_c, n);
    } else {
        constexpr int TILE = 32;
        dim3 block(TILE, TILE);
        dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
        BlockGemmKernel<TILE><<<grid, block>>>(d_a, d_b, d_c, n);
    }

    std::vector<float> c(total_elements);

    cudaMemcpy(c.data(), d_c, matrix_bytes, cudaMemcpyDeviceToHost);

    return c;
}