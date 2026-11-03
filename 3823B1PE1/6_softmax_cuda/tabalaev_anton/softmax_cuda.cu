#include "softmax_cuda.h"
#include <cuda_runtime.h>
#include <cfloat>
#include <vector>

namespace
{

   constexpr int BLOCK_SIZE = 256;

   __global__ void SoftmaxRowKernel(float *__restrict__ data, int row_size)
   {
      __shared__ float s_reduce[BLOCK_SIZE];

      const int tid = threadIdx.x;
      float *row_ptr = data + static_cast<size_t>(blockIdx.x) * row_size;
      const int vec_cols = row_size / 4;
      float4 *row_vec = reinterpret_cast<float4 *>(row_ptr);

      float local_max = -FLT_MAX;
      for (int i = tid; i < vec_cols; i += BLOCK_SIZE)
      {
         const float4 v = row_vec[i];
         local_max = fmaxf(local_max, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
      }
      for (int i = vec_cols * 4 + tid; i < row_size; i += BLOCK_SIZE)
      {
         local_max = fmaxf(local_max, row_ptr[i]);
      }

      s_reduce[tid] = local_max;
      __syncthreads();

#pragma unroll
      for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1)
      {
         if (tid < stride)
         {
            s_reduce[tid] = fmaxf(s_reduce[tid], s_reduce[tid + stride]);
         }
         __syncthreads();
      }
      const float row_max = s_reduce[0];

      float local_sum = 0.0f;
      for (int i = tid; i < vec_cols; i += BLOCK_SIZE)
      {
         const float4 v = row_vec[i];
         local_sum += __expf(v.x - row_max) + __expf(v.y - row_max) +
                      __expf(v.z - row_max) + __expf(v.w - row_max);
      }
      for (int i = vec_cols * 4 + tid; i < row_size; i += BLOCK_SIZE)
      {
         local_sum += __expf(row_ptr[i] - row_max);
      }

      s_reduce[tid] = local_sum;
      __syncthreads();

#pragma unroll
      for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1)
      {
         if (tid < stride)
         {
            s_reduce[tid] += s_reduce[tid + stride];
         }
         __syncthreads();
      }
      const float inv_sum = __fdividef(1.0f, s_reduce[0]);

      for (int i = tid; i < vec_cols; i += BLOCK_SIZE)
      {
         float4 v = row_vec[i];
         v.x = __expf(v.x - row_max) * inv_sum;
         v.y = __expf(v.y - row_max) * inv_sum;
         v.z = __expf(v.z - row_max) * inv_sum;
         v.w = __expf(v.w - row_max) * inv_sum;
         row_vec[i] = v;
      }
      for (int i = vec_cols * 4 + tid; i < row_size; i += BLOCK_SIZE)
      {
         row_ptr[i] = __expf(row_ptr[i] - row_max) * inv_sum;
      }
   }

   static float *g_d_data = nullptr;
   static size_t g_allocated_elements = 0;
   static cudaStream_t g_stream = nullptr;

   void ensure_softmax_resources(size_t total_elements)
   {
      if (!g_stream)
      {
         cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking);
      }
      if (total_elements > g_allocated_elements)
      {
         if (g_d_data)
         {
            cudaFree(g_d_data);
         }
         cudaMalloc(&g_d_data, total_elements * sizeof(float));
         g_allocated_elements = total_elements;
      }
   }

} // namespace

std::vector<float> SoftmaxCUDA(const std::vector<float> &input, int row_count)
{
   if (input.empty() || row_count <= 0)
   {
      return {};
   }

   const size_t total_elements = input.size();
   const int row_size = static_cast<int>(total_elements / row_count);
   const size_t bytes = total_elements * sizeof(float);

   ensure_softmax_resources(total_elements);

   cudaMemcpyAsync(g_d_data, input.data(), bytes, cudaMemcpyHostToDevice, g_stream);

   SoftmaxRowKernel<<<row_count, BLOCK_SIZE, 0, g_stream>>>(g_d_data, row_size);

   std::vector<float> output(total_elements);

   cudaMemcpyAsync(output.data(), g_d_data, bytes, cudaMemcpyDeviceToHost, g_stream);
   cudaStreamSynchronize(g_stream);

   return output;
}
