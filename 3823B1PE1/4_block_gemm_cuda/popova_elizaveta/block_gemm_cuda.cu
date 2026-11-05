#include "block_gemm_cuda.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstddef>

namespace
{
    constexpr int TILE = 32;

    __global__ void BlockGemmKernel(const float *__restrict__ A,
                                    const float *__restrict__ B,
                                    float *__restrict__ C,
                                    int n)
    {
        __shared__ float As[TILE][TILE];
        __shared__ float Bs[TILE][TILE];

        const int tx = threadIdx.x;
        const int ty = threadIdx.y;

        const int blockRow = blockIdx.y * TILE;
        const int blockCol = blockIdx.x * TILE;

        float c00 = 0.f, c01 = 0.f, c10 = 0.f, c11 = 0.f;

        const int numTiles = n / TILE;
        for (int t = 0; t < numTiles; ++t)
        {
            const int gk = t * TILE;

            As[ty][tx] = A[(blockRow + ty) * n + gk + tx];
            As[ty][tx + 16] = A[(blockRow + ty) * n + gk + tx + 16];
            As[ty + 16][tx] = A[(blockRow + ty + 16) * n + gk + tx];
            As[ty + 16][tx + 16] = A[(blockRow + ty + 16) * n + gk + tx + 16];

            Bs[ty][tx] = B[(gk + ty) * n + blockCol + tx];
            Bs[ty][tx + 16] = B[(gk + ty) * n + blockCol + tx + 16];
            Bs[ty + 16][tx] = B[(gk + ty + 16) * n + blockCol + tx];
            Bs[ty + 16][tx + 16] = B[(gk + ty + 16) * n + blockCol + tx + 16];

            __syncthreads();

#pragma unroll
            for (int k = 0; k < TILE; ++k)
            {
                const float a0 = As[ty * 2][k];
                const float a1 = As[ty * 2 + 1][k];
                const float b0 = Bs[k][tx * 2];
                const float b1 = Bs[k][tx * 2 + 1];

                c00 += a0 * b0;
                c01 += a0 * b1;
                c10 += a1 * b0;
                c11 += a1 * b1;
            }

            __syncthreads();
        }

        C[(blockRow + ty * 2) * n + blockCol + tx * 2] = c00;
        C[(blockRow + ty * 2) * n + blockCol + tx * 2 + 1] = c01;
        C[(blockRow + ty * 2 + 1) * n + blockCol + tx * 2] = c10;
        C[(blockRow + ty * 2 + 1) * n + blockCol + tx * 2 + 1] = c11;
    }

    struct Buffers
    {
        float *d_a = nullptr;
        float *d_b = nullptr;
        float *d_c = nullptr;
        std::size_t cap = 0;

        ~Buffers()
        {
            if (d_a)
                cudaFree(d_a);
            if (d_b)
                cudaFree(d_b);
            if (d_c)
                cudaFree(d_c);
        }

        bool ensure(std::size_t elems)
        {
            if (elems <= cap)
                return true;
            if (d_a)
            {
                cudaFree(d_a);
                d_a = nullptr;
            }
            if (d_b)
            {
                cudaFree(d_b);
                d_b = nullptr;
            }
            if (d_c)
            {
                cudaFree(d_c);
                d_c = nullptr;
            }
            const std::size_t bytes = elems * sizeof(float);
            if (cudaMalloc(&d_a, bytes) != cudaSuccess)
                return false;
            if (cudaMalloc(&d_b, bytes) != cudaSuccess)
                return false;
            if (cudaMalloc(&d_c, bytes) != cudaSuccess)
                return false;
            cap = elems;
            return true;
        }
    };

    Buffers g_buf;

} // namespace

std::vector<float> BlockGemmCUDA(const std::vector<float> &a,
                                 const std::vector<float> &b,
                                 int n)
{
    std::vector<float> c(static_cast<std::size_t>(n) * n, 0.0f);
    if (n <= 0)
        return c;

    const std::size_t elems = static_cast<std::size_t>(n) * n;
    const std::size_t bytes = elems * sizeof(float);

    if (!g_buf.ensure(elems))
    {
        std::fprintf(stderr, "CUDA: alloc failed\n");
        return c;
    }

    cudaMemcpy(g_buf.d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(g_buf.d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid(n / TILE, n / TILE);

    BlockGemmKernel<<<grid, block>>>(g_buf.d_a, g_buf.d_b, g_buf.d_c, n);

    cudaMemcpy(c.data(), g_buf.d_c, bytes, cudaMemcpyDeviceToHost);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }

    return c;
}