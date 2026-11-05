#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>

namespace
{

   constexpr int BLOCK_DIM_X = 32;
   constexpr int BLOCK_DIM_Y = 8;
   constexpr int VEC_SIZE = 4;

   __global__ void NaiveGemmKernel(const float *__restrict__ a,
                                   const float *__restrict__ b,
                                   float *__restrict__ c,
                                   int n)
   {
      const int col_vec = blockIdx.x * blockDim.x + threadIdx.x;
      const int row = blockIdx.y * blockDim.y + threadIdx.y;
      const int col = col_vec * VEC_SIZE;

      if (row >= n || col >= n)
      {
         return;
      }

      float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      const int n_vec = n / VEC_SIZE;
      const float4 *b_vec_ptr = reinterpret_cast<const float4 *>(b);

#pragma unroll 4
      for (int k = 0; k < n; ++k)
      {
         float a_val = a[row * n + k];

         float4 b_val = b_vec_ptr[k * n_vec + col_vec];

         acc.x += a_val * b_val.x;
         acc.y += a_val * b_val.y;
         acc.z += a_val * b_val.z;
         acc.w += a_val * b_val.w;
      }

      reinterpret_cast<float4 *>(c)[row * n_vec + col_vec] = acc;
   }

   __global__ void NaiveGemmScalarKernel(const float *__restrict__ a,
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

   void ensure_device_buffers(size_t total_elements)
   {
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

std::vector<float> NaiveGemmCUDA(const std::vector<float> &a,
                                 const std::vector<float> &b,
                                 int n)
{
   if (n <= 0)
   {
      return {};
   }

   const size_t total_elements = static_cast<size_t>(n) * n;
   const size_t bytes = total_elements * sizeof(float);

   ensure_device_buffers(total_elements);

   cudaMemcpy(g_d_a, a.data(), bytes, cudaMemcpyHostToDevice);
   cudaMemcpy(g_d_b, b.data(), bytes, cudaMemcpyHostToDevice);

   dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);

   if (n >= VEC_SIZE && (n % VEC_SIZE == 0))
   {
      dim3 grid((n / VEC_SIZE + block.x - 1) / block.x,
                (n + block.y - 1) / block.y);
      NaiveGemmKernel<<<grid, block>>>(g_d_a, g_d_b, g_d_c, n);
   }
   else
   {
      dim3 grid((n + block.x - 1) / block.x,
                (n + block.y - 1) / block.y);
      NaiveGemmScalarKernel<<<grid, block>>>(g_d_a, g_d_b, g_d_c, n);
   }

   std::vector<float> c(total_elements);
   cudaMemcpy(c.data(), g_d_c, bytes, cudaMemcpyDeviceToHost);

   return c;
}
