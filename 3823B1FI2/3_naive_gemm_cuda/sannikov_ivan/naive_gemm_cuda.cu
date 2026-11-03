#include <cuda_runtime.h>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include "naive_gemm_cuda.h"

namespace {
__global__ void SmallGemmKernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n) {
  const int col = threadIdx.x;
  const int row = threadIdx.y;
  if (row >= n || col >= n) {
    return;
  }
  float sum = 0.0f;
  for (int k = 0; k < n; ++k) {
    sum = fmaf(a[row * n + k], b[k * n + col], sum);
  }
  c[row * n + col] = sum;
}


__global__ void VectorGemmKernel(const float4* __restrict__ a, const float4* __restrict__ b,float4* __restrict__ c, int n) {
  const int col4 = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = 4 * (blockIdx.y * blockDim.y + threadIdx.y);
  const int n4 = n / 4;
  if (row >= n || col4 >= n4) {
    return;
  }

  float4 sums[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    sums[r] = make_float4(0, 0, 0, 0);
  }

  for (int k4 = 0; k4 < n4; ++k4) {
    const std::size_t k = 4 * static_cast<std::size_t>(k4);
    const float4 b0 = b[(k + 0) * n4 + col4];
    const float4 b1 = b[(k + 1) * n4 + col4];
    const float4 b2 = b[(k + 2) * n4 + col4];
    const float4 b3 = b[(k + 3) * n4 + col4];

#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const float4 av = a[static_cast<std::size_t>(row + r) * n4 + k4];
      sums[r].x = fmaf(av.x, b0.x, sums[r].x);
      sums[r].y = fmaf(av.x, b0.y, sums[r].y);
      sums[r].z = fmaf(av.x, b0.z, sums[r].z);
      sums[r].w = fmaf(av.x, b0.w, sums[r].w);
      sums[r].x = fmaf(av.y, b1.x, sums[r].x);
      sums[r].y = fmaf(av.y, b1.y, sums[r].y);
      sums[r].z = fmaf(av.y, b1.z, sums[r].z);
      sums[r].w = fmaf(av.y, b1.w, sums[r].w);
      sums[r].x = fmaf(av.z, b2.x, sums[r].x);
      sums[r].y = fmaf(av.z, b2.y, sums[r].y);
      sums[r].z = fmaf(av.z, b2.z, sums[r].z);
      sums[r].w = fmaf(av.z, b2.w, sums[r].w);
      sums[r].x = fmaf(av.w, b3.x, sums[r].x);
      sums[r].y = fmaf(av.w, b3.y, sums[r].y);
      sums[r].z = fmaf(av.w, b3.z, sums[r].z);
      sums[r].w = fmaf(av.w, b3.w, sums[r].w);
    }
  }

#pragma unroll
  for (int r = 0; r < 4; ++r) {
    c[static_cast<std::size_t>(row + r) * n4 + col4] = sums[r];
  }
}
}  // namespace

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
  if (n <= 0 || (n & (n - 1)) != 0) {
    throw std::invalid_argument("n must be a positive power of two");
  }
  const std::size_t side = static_cast<std::size_t>(n);
  if (side > std::numeric_limits<std::size_t>::max() / side / (3 * sizeof(float))) {
    throw std::length_error("Matrix size overflow");
  }
  const std::size_t count = side * side;
  if (a.size() != count || b.size() != count) {
    throw std::invalid_argument("Expected n*n elements in both matrices");
  }
  const std::size_t bytes = count * sizeof(float);
  float* memory = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&memory), 3 * bytes);
  float* d_a = memory;
  float* d_b = d_a + count;
  float* d_c = d_b + count;
  cudaMemcpyAsync(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

  if (n < 4) {
    SmallGemmKernel<<<1, dim3(2, 2)>>>(d_a, d_b, d_c, n);
  } else {
    const dim3 block(32, 4);
    const unsigned groups = static_cast<unsigned>(n / 4);
    const dim3 grid((groups + block.x - 1) / block.x, (groups + block.y - 1) / block.y);
    VectorGemmKernel<<<grid, block>>>(reinterpret_cast<const float4*>(d_a), reinterpret_cast<const float4*>(d_b), reinterpret_cast<float4*>(d_c), n);
  }
  std::vector<float> c(count);
  cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
  cudaFree(memory);
  return c;
}
