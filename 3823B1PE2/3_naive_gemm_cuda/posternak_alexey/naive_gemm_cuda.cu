#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <cstdlib>

#define BLOCK_SIZE 16
#define TILE_SIZE 4 // каждый поток считает 4*4 = 16 элементов С

__global__ void naive_gemm_kernel(const float* __restrict__ a,
                                   const float* __restrict__ b,
                                   float* __restrict__ c,
                                   int n) {
    // говорим потоку, какую часть ему вычислять
    int row = blockIdx.y * BLOCK_SIZE * TILE_SIZE + threadIdx.y * TILE_SIZE;
    int col = blockIdx.x * BLOCK_SIZE * TILE_SIZE + threadIdx.x * TILE_SIZE;
    
    // частичные суммы
    float sum[TILE_SIZE][TILE_SIZE];
    #pragma unroll
    for (int i = 0; i < TILE_SIZE; ++i) {
        #pragma unroll
        for (int j = 0; j < TILE_SIZE; ++j) {
            sum[i][j] = 0.0f;
        }
    }
    
    for (int k = 0; k < n; ++k) {
        // загружаем столбец части матрицы
        float a_val[TILE_SIZE];
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            if (row + i < n) {
                a_val[i] = a[(row + i) * n + k];
            } else {
                a_val[i] = 0.0f;
            }
        }
        
        // загружаем строку
        float b_val[TILE_SIZE];
        #pragma unroll
        for (int j = 0; j < TILE_SIZE; ++j) {
            if (col + j < n) {
                b_val[j] = b[k * n + (col + j)];
            } else {
                b_val[j] = 0.0f;
            }
        }
        
        // умножаем
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            #pragma unroll
            for (int j = 0; j < TILE_SIZE; ++j) {
                sum[i][j] += a_val[i] * b_val[j];
            }
        }
    }

    // записываем
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

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                  const std::vector<float>& b,
                                  int n) {
    if (n == 0) return std::vector<float>();
    
    size_t size = static_cast<size_t>(n) * n;
    
    if (n > allocated_n) {  // перевыеляем память только если пришел больший размер
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
    
    naive_gemm_kernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
    
    std::vector<float> c(size);
    
    cudaMemcpyAsync(c.data(), d_c, size * sizeof(float), cudaMemcpyDeviceToHost, 0);
    
    cudaDeviceSynchronize();
    
    return c;
}
