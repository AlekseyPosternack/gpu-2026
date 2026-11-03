#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>

namespace
{

   static cublasHandle_t g_handle = nullptr;
   static cudaStream_t g_stream = nullptr;

   static float *g_d_a = nullptr;
   static float *g_d_b = nullptr;
   static float *g_d_c = nullptr;
   static size_t g_allocated_elements = 0;

   void ensure_cublas_resources(size_t total_elements)
   {
      if (!g_handle)
      {
         cublasCreate(&g_handle);
         cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking);
         cublasSetStream(g_handle, g_stream);
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

std::vector<float> GemmCUBLAS(const std::vector<float> &a,
                              const std::vector<float> &b,
                              int n)
{
   if (n <= 0)
   {
      return {};
   }

   const size_t total_elements = static_cast<size_t>(n) * n;
   const size_t bytes = total_elements * sizeof(float);

   ensure_cublas_resources(total_elements);

   cudaMemcpyAsync(g_d_a, a.data(), bytes, cudaMemcpyHostToDevice, g_stream);
   cudaMemcpyAsync(g_d_b, b.data(), bytes, cudaMemcpyHostToDevice, g_stream);

   constexpr float alpha = 1.0f;
   constexpr float beta = 0.0f;

   cublasSgemm(g_handle,
               CUBLAS_OP_N, CUBLAS_OP_N,
               n, n, n,
               &alpha,
               g_d_b, n,
               g_d_a, n,
               &beta,
               g_d_c, n);

   std::vector<float> c(total_elements);

   cudaMemcpyAsync(c.data(), g_d_c, bytes, cudaMemcpyDeviceToHost, g_stream);
   cudaStreamSynchronize(g_stream);

   return c;
}
