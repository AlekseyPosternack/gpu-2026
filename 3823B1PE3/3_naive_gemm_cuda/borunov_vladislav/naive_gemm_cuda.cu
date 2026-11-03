// Copyright 2026 Borunov Vladislav

#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>
#include <stdexcept>

namespace {


constexpr int ROWS_PER_THREAD = 8;
constexpr int COLS_PER_THREAD = 4;


constexpr int BLOCK_DIM_X = 32;
constexpr int BLOCK_DIM_Y = 4;

inline void CheckCUDA(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}


struct DeviceBuffers {
  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;

  ~DeviceBuffers() {
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
  }
};


__global__ void SmallGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c, int n) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= n || col >= n) return;

  float sum = 0.0f;
  for (int k = 0; k < n; ++k) {
    sum = fmaf(a[row * n + k], b[k * n + col], sum);
  }
  c[row * n + col] = sum;
}


__global__ void NaiveGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c, int n) {
  int first_col = (blockIdx.x * BLOCK_DIM_X + threadIdx.x) * COLS_PER_THREAD;
  int first_row = (blockIdx.y * BLOCK_DIM_Y + threadIdx.y) * ROWS_PER_THREAD;

  
  float sums[ROWS_PER_THREAD][COLS_PER_THREAD] = {};

  for (int k = 0; k < n; ++k) {
    float4 b_vals =
        *reinterpret_cast<const float4*>(b + k * n + first_col);

#pragma unroll
    for (int dr = 0; dr < ROWS_PER_THREAD; ++dr) {
      float a_val = a[(first_row + dr) * n + k];
      sums[dr][0] = fmaf(a_val, b_vals.x, sums[dr][0]);
      sums[dr][1] = fmaf(a_val, b_vals.y, sums[dr][1]);
      sums[dr][2] = fmaf(a_val, b_vals.z, sums[dr][2]);
      sums[dr][3] = fmaf(a_val, b_vals.w, sums[dr][3]);
    }
  }


#pragma unroll
  for (int dr = 0; dr < ROWS_PER_THREAD; ++dr) {
    *reinterpret_cast<float4*>(c + (first_row + dr) * n + first_col) =
        make_float4(sums[dr][0], sums[dr][1], sums[dr][2], sums[dr][3]);
  }
}

}  

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b, int n) {
  const size_t elem_count = static_cast<size_t>(n) * n;
  const size_t byte_count = elem_count * sizeof(float);

  DeviceBuffers dev;
  CheckCUDA(cudaMalloc(&dev.d_a, byte_count));
  CheckCUDA(cudaMalloc(&dev.d_b, byte_count));
  CheckCUDA(cudaMalloc(&dev.d_c, byte_count));

  CheckCUDA(
      cudaMemcpy(dev.d_a, a.data(), byte_count, cudaMemcpyHostToDevice));
  CheckCUDA(
      cudaMemcpy(dev.d_b, b.data(), byte_count, cudaMemcpyHostToDevice));

  dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);

  if (n < 128) {
    dim3 grid((n + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
              (n + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
    SmallGemmKernel<<<grid, block>>>(dev.d_a, dev.d_b, dev.d_c, n);
  } else {
    dim3 grid(n / (BLOCK_DIM_X * COLS_PER_THREAD),
              n / (BLOCK_DIM_Y * ROWS_PER_THREAD));
    NaiveGemmKernel<<<grid, block>>>(dev.d_a, dev.d_b, dev.d_c, n);
  }
  CheckCUDA(cudaGetLastError());

  std::vector<float> c(elem_count);
  CheckCUDA(cudaMemcpy(c.data(), dev.d_c, byte_count, cudaMemcpyDeviceToHost));
  return c;
}
