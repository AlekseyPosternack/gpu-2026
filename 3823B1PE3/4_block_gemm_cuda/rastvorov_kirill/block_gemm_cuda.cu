#include "block_gemm_cuda.h"

#include <cuda_runtime.h>

namespace {

constexpr int kTile = 128;
constexpr int kDepth = 16;
constexpr int kThreads = 256;
constexpr int kSmallTile = 32;

//------------------------------------------------------------------
//function: BlockGemmKernel
//purpose: блочное умножение матриц на GPU: блок потоков считает тайл C 128x128 через shared memory, каждый поток - 8x8 элементов
//------------------------------------------------------------------
__global__ void __launch_bounds__(kThreads)
BlockGemmKernel(const float* __restrict__ a, const float* __restrict__ b,
                float* __restrict__ c, int n) {
    __shared__ __align__(16) float block_a[kDepth][kTile];
    __shared__ __align__(16) float block_b[kDepth][kTile];

    const int tid = threadIdx.x;
    const int tx = tid % 16;
    const int ty = tid / 16;
    const int row0 = blockIdx.y * kTile;
    const int col0 = blockIdx.x * kTile;

    float acc[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = 0.0f;
    }

    for (int k0 = 0; k0 < n; k0 += kDepth) {
#pragma unroll
        for (int i = tid; i < kTile * kDepth / 4; i += kThreads) {
            const int a_row = i / (kDepth / 4);
            const int a_col = i % (kDepth / 4) * 4;
            const float4 av = *reinterpret_cast<const float4*>(
                a + static_cast<size_t>(row0 + a_row) * n + k0 + a_col);
            block_a[a_col + 0][a_row] = av.x;
            block_a[a_col + 1][a_row] = av.y;
            block_a[a_col + 2][a_row] = av.z;
            block_a[a_col + 3][a_row] = av.w;

            const int b_row = i / (kTile / 4);
            const int b_col = i % (kTile / 4) * 4;
            *reinterpret_cast<float4*>(&block_b[b_row][b_col]) =
                *reinterpret_cast<const float4*>(
                    b + static_cast<size_t>(k0 + b_row) * n + col0 + b_col);
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < kDepth; ++k) {
            const float4 a0 = *reinterpret_cast<const float4*>(&block_a[k][ty * 4]);
            const float4 a1 = *reinterpret_cast<const float4*>(&block_a[k][64 + ty * 4]);
            const float4 b0 = *reinterpret_cast<const float4*>(&block_b[k][tx * 4]);
            const float4 b1 = *reinterpret_cast<const float4*>(&block_b[k][64 + tx * 4]);
            const float as[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float bs[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
            for (int i = 0; i < 8; ++i) {
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[i][j] += as[i] * bs[j];
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = row0 + (i < 4 ? ty * 4 + i : 64 + ty * 4 + i - 4);
        float* c_row = c + static_cast<size_t>(row) * n + col0;
        *reinterpret_cast<float4*>(c_row + tx * 4) =
            make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        *reinterpret_cast<float4*>(c_row + 64 + tx * 4) =
            make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
    }
}

//------------------------------------------------------------------
//function: BlockGemmSmallKernel
//purpose: блочное умножение для маленьких матриц (n < 128): тайлы 32x32 в shared memory, один поток - один элемент C
//------------------------------------------------------------------
__global__ void BlockGemmSmallKernel(const float* __restrict__ a,
                                     const float* __restrict__ b,
                                     float* __restrict__ c, int n) {
    __shared__ float tile_a[kSmallTile][kSmallTile];
    __shared__ float tile_b[kSmallTile][kSmallTile];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * kSmallTile + ty;
    const int col = blockIdx.x * kSmallTile + tx;

    float sum = 0.0f;
    for (int t = 0; t < n; t += kSmallTile) {
        tile_a[ty][tx] = (row < n && t + tx < n) ? a[row * n + t + tx] : 0.0f;
        tile_b[ty][tx] = (t + ty < n && col < n) ? b[(t + ty) * n + col] : 0.0f;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < kSmallTile; ++k) {
            sum += tile_a[ty][k] * tile_b[k][tx];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

float* g_buffer = nullptr;
size_t g_capacity = 0;

//------------------------------------------------------------------
//function: EnsureCapacity
//purpose: выделяет видеопамять под A, B и C один раз и переиспользует её между вызовами, пока хватает размера
//------------------------------------------------------------------
void EnsureCapacity(size_t elements) {
    if (elements <= g_capacity) return;
    if (g_buffer) cudaFree(g_buffer);
    cudaMalloc(&g_buffer, 3 * elements * sizeof(float));
    g_capacity = elements;
}

}

//------------------------------------------------------------------
//function: BlockGemmCUDA
//purpose: умножает матрицы A и B (n x n, по строкам) на GPU: копирует данные, запускает нужное ядро и возвращает C
//------------------------------------------------------------------
std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) return {};
    const size_t elements = static_cast<size_t>(n) * n;
    const size_t bytes = elements * sizeof(float);

    EnsureCapacity(elements);
    float* d_a = g_buffer;
    float* d_b = g_buffer + elements;
    float* d_c = g_buffer + 2 * elements;

    cudaMemcpyAsync(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    if (n % kTile == 0) {
        const dim3 grid(n / kTile, n / kTile);
        BlockGemmKernel<<<grid, kThreads>>>(d_a, d_b, d_c, n);
    } else {
        const dim3 block(kSmallTile, kSmallTile);
        const dim3 grid((n + kSmallTile - 1) / kSmallTile,
                        (n + kSmallTile - 1) / kSmallTile);
        BlockGemmSmallKernel<<<grid, block>>>(d_a, d_b, d_c, n);
    }

    std::vector<float> c(elements);
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    return c;
}
