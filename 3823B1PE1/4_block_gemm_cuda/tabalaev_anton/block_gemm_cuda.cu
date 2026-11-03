#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>

namespace
{

   constexpr int TILE_SIZE = 32;
   constexpr int BLOCK_DIM_X = 32;
   constexpr int BLOCK_DIM_Y = 8;
   constexpr int ITEMS_PER_THREAD = TILE_SIZE / BLOCK_DIM_Y;

   __global__ void BlockGemmKernel(const float *__restrict__ a,
                                   const float *__restrict__ b,
                                   float *__restrict__ c,
                                   int n)
   {
      __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
      __shared__ float tile_b[TILE_SIZE][TILE_SIZE];

      const int tx = threadIdx.x;
      const int ty = threadIdx.y;

      const int block_row_base = blockIdx.y * TILE_SIZE;
      const int block_col_base = blockIdx.x * TILE_SIZE;

      const int global_col = block_col_base + tx;
      const int local_row_start = ty * ITEMS_PER_THREAD;
      const int global_row_start = block_row_base + local_row_start;

      float acc[ITEMS_PER_THREAD] = {0.0f};

      for (int k_step = 0; k_step < n; k_step += TILE_SIZE)
      {
#pragma unroll
         for (int i = 0; i < ITEMS_PER_THREAD; ++i)
         {
            tile_a[local_row_start + i][tx] = a[(global_row_start + i) * n + (k_step + tx)];
            tile_b[local_row_start + i][tx] = b[(k_step + local_row_start + i) * n + global_col];
         }

         __syncthreads();

#pragma unroll
         for (int k = 0; k < TILE_SIZE; ++k)
         {
            const float b_val = tile_b[k][tx];
#pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD; ++i)
            {
               acc[i] = fmaf(tile_a[local_row_start + i][k], b_val, acc[i]);
            }
         }

         __syncthreads();
      }

#pragma unroll
      for (int i = 0; i < ITEMS_PER_THREAD; ++i)
      {
         c[(global_row_start + i) * n + global_col] = acc[i];
      }
   }

   __global__ void BlockGemmSmallKernel(const float *__restrict__ a,
                                        const float *__restrict__ b,
                                        float *__restrict__ c,
                                        int n)
   {
      const int col = blockIdx.x * blockDim.x + threadIdx.x;
      const int row = blockIdx.y * blockDim.y + threadIdx.y;

      if (row < n && col < n)
      {
         float sum = 0.0f;
         for (int k = 0; k < n; ++k)
         {
            sum += a[row * n + k] * b[k * n + col];
         }
         c[row * n + col] = sum;
      }
   }

   static float *g_d_a = nullptr;
   static float *g_d_b = nullptr;
   static float *g_d_c = nullptr;
   static size_t g_allocated_elements = 0;
   static cudaStream_t g_stream = nullptr;

   void ensure_device_resources(size_t total_elements)
   {
      if (!g_stream)
      {
         cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking);
      }
      if (total_elements > g_allocated_elements)
      {
         if (g_d_a)
            cudaFree(g_d_a);
         if (g_d_b)
            cudaFree(g_d_b);
         if (g_d_c)
            cudaFree(g_d_c);

         const size_t bytes = total_elements * sizeof(float);
         cudaMalloc(&g_d_a, bytes);
         cudaMalloc(&g_d_b, bytes);
         cudaMalloc(&g_d_c, bytes);
         g_allocated_elements = total_elements;
      }
   }

} // namespace

std::vector<float> BlockGemmCUDA(const std::vector<float> &a,
                                 const std::vector<float> &b,
                                 int n)
{
   if (n <= 0)
   {
      return {};
   }

   const size_t total_elements = static_cast<size_t>(n) * n;
   const size_t bytes = total_elements * sizeof(float);

   ensure_device_resources(total_elements);

   cudaMemcpyAsync(g_d_a, a.data(), bytes, cudaMemcpyHostToDevice, g_stream);
   cudaMemcpyAsync(g_d_b, b.data(), bytes, cudaMemcpyHostToDevice, g_stream);

   if (n >= TILE_SIZE)
   {
      dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
      dim3 grid(n / TILE_SIZE, n / TILE_SIZE);
      BlockGemmKernel<<<grid, block, 0, g_stream>>>(g_d_a, g_d_b, g_d_c, n);
   }
   else
   {
      dim3 block(16, 16);
      dim3 grid((n + 15) / 16, (n + 15) / 16);
      BlockGemmSmallKernel<<<grid, block, 0, g_stream>>>(g_d_a, g_d_b, g_d_c, n);
   }

   std::vector<float> c(total_elements);

   cudaMemcpyAsync(c.data(), g_d_c, bytes, cudaMemcpyDeviceToHost, g_stream);
   cudaStreamSynchronize(g_stream);

   return c;
}
