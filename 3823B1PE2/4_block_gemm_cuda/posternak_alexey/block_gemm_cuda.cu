#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <cstdlib>

#define BLOCK_SIZE 16
#define TILE_SIZE 2

__global__ void block_gemm_kernel(const float* __restrict__ a,
                                   const float* __restrict__ b,
                                   float* __restrict__ c,
                                   int n) {
    __shared__ float As[BLOCK_SIZE * TILE_SIZE][BLOCK_SIZE + 1];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE * TILE_SIZE + 1];
    
    // координаты блока в сетке
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    
    // координаты потока внутри блока
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = block_row * BLOCK_SIZE * TILE_SIZE + ty * TILE_SIZE;
    int col = block_col * BLOCK_SIZE * TILE_SIZE + tx * TILE_SIZE;
    
    float sum[TILE_SIZE][TILE_SIZE];
    #pragma unroll
    for (int i = 0; i < TILE_SIZE; ++i) {
        #pragma unroll
        for (int j = 0; j < TILE_SIZE; ++j) {
            sum[i][j] = 0.0f;
        }
    }
    
    // кол-во тайлов в строке/столбце
    int num_tiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (int t = 0; t < num_tiles; ++t) {
        // определяем координаты
        int tile_row = t * BLOCK_SIZE + ty;
        int tile_col = t * BLOCK_SIZE + tx;
        
        // загрузка блока A в sm
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            int a_row = row + i;
            int a_col = tile_col;
            if (a_row < n && a_col < n) {
                As[ty * TILE_SIZE + i][tx] = a[a_row * n + a_col];
            } else {
                As[ty * TILE_SIZE + i][tx] = 0.0f;
            }
        }
        
        // загрузка блока B в sm
        #pragma unroll
        for (int j = 0; j < TILE_SIZE; ++j) {
            int b_row = tile_row;
            int b_col = col + j;
            if (b_row < n && b_col < n) {
                Bs[ty][tx * TILE_SIZE + j] = b[b_row * n + b_col];
            } else {
                Bs[ty][tx * TILE_SIZE + j] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // умножение блоков: As * Bs и их запись в sum
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            #pragma unroll
            for (int i = 0; i < TILE_SIZE; ++i) {
                #pragma unroll
                for (int j = 0; j < TILE_SIZE; ++j) {
                    sum[i][j] += As[ty * TILE_SIZE + i][k] * Bs[k][tx * TILE_SIZE + j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // запись результата в global memory
    #pragma unroll
    for (int i = 0; i < TILE_SIZE; ++i) {
        if (row + i < n) {
            #pragma unroll
            for (int j = 0; j < TILE_SIZE; ++j) {
                if (col + j < n) {
                    c[(row + i) * n + (col + j)] = sum[i][j];
                }
            }
        }
    }
}

// device memory
static float* d_a = nullptr;
static float* d_b = nullptr;
static float* d_c = nullptr;
static size_t allocated_n = 0;

static void cleanup_cuda_resources() {
    if (d_a) cudaFree(d_a);
    if (d_b) cudaFree(d_b);
    if (d_c) cudaFree(d_c);
    d_a = nullptr;
    d_b = nullptr;
    d_c = nullptr;
    allocated_n = 0;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                  const std::vector<float>& b,
                                  int n) {
    if (n == 0) return std::vector<float>();
    
    size_t size = static_cast<size_t>(n) * n;
    
    if (n > allocated_n) {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
        
        cudaMalloc(&d_a, size * sizeof(float));
        cudaMalloc(&d_b, size * sizeof(float));
        cudaMalloc(&d_c, size * sizeof(float));
        allocated_n = n;
        
        static bool cleanup_registered = false;
        if (!cleanup_registered) {
            std::atexit(cleanup_cuda_resources);
            cleanup_registered = true;
        }
    }
    
    cudaMemcpyAsync(d_a, a.data(), size * sizeof(float), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_b, b.data(), size * sizeof(float), cudaMemcpyHostToDevice, 0);
    
    // block size selection
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((n + BLOCK_SIZE * TILE_SIZE - 1) / (BLOCK_SIZE * TILE_SIZE),
                (n + BLOCK_SIZE * TILE_SIZE - 1) / (BLOCK_SIZE * TILE_SIZE));
    
    block_gemm_kernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
    
    std::vector<float> c(size);
    
    cudaMemcpyAsync(c.data(), d_c, size * sizeof(float), cudaMemcpyDeviceToHost, 0);
    
    cudaDeviceSynchronize();
    
    return c;
}
