#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <memory>
#include <stdexcept>

namespace {

constexpr int kBlockSize = 16;

__global__ void block_gemm_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n) {
  __shared__ float a_tile[kBlockSize][kBlockSize];
  __shared__ float b_tile[kBlockSize][kBlockSize];
  __shared__ float c_tile[kBlockSize][kBlockSize];
  const int thr_x = threadIdx.x;
  const int thr_y = threadIdx.y;
  const int row = blockIdx.y * kBlockSize + thr_y;
  const int col = blockIdx.x * kBlockSize + thr_x;
  const std::size_t row_off = static_cast<std::size_t>(row) * n;
  c_tile[thr_y][thr_x] = 0.0f;

  for (int tile_index = 0; tile_index <= (n - 1) / kBlockSize; ++tile_index) {
    const int a_col = tile_index * kBlockSize + thr_x;
    const int b_row = tile_index * kBlockSize + thr_y;

    a_tile[thr_y][thr_x] = row < n && a_col < n ? a[row_off + a_col] : 0.0f;
    b_tile[thr_y][thr_x] = b_row < n && col < n ? b[static_cast<std::size_t>(b_row) * n + col] : 0.0f;
    __syncthreads();

    float sum = c_tile[thr_y][thr_x];
#pragma unroll
    for (int k = 0; k < kBlockSize; ++k) {
      sum += a_tile[thr_y][k] * b_tile[k][thr_x];
    }
    c_tile[thr_y][thr_x] = sum;
    __syncthreads();
  }

  if (row < n && col < n) { c[row_off + col] = c_tile[thr_y][thr_x]; }
}

using DeviceBuffer = std::unique_ptr<float, decltype(&cudaFree)>;
DeviceBuffer allocate_device_buffer(std::size_t byte_count) {
  float* buffer = nullptr;
  cudaMalloc(&buffer, byte_count);
  return DeviceBuffer(buffer, cudaFree);
}
}  // namespace

std::vector<float> BlockGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
  if (n < 0) {
    throw std::invalid_argument("nonnegative size error");
  }
  const std::size_t element_count = static_cast<std::size_t>(n) * n;
  if (a.size() != element_count || b.size() != element_count) {
    throw std::invalid_argument("size error");
  }
  if (element_count == 0) {
    return {};
  }

  std::vector<float> c(element_count);
  const std::size_t byte_count = element_count * sizeof(float);

  DeviceBuffer device_a = allocate_device_buffer(byte_count);
  DeviceBuffer device_b = allocate_device_buffer(byte_count);
  DeviceBuffer device_c = allocate_device_buffer(byte_count);
  cudaMemcpy(device_a.get(), a.data(), byte_count, cudaMemcpyHostToDevice);
  cudaMemcpy(device_b.get(), b.data(), byte_count, cudaMemcpyHostToDevice);
  const dim3 block_size(kBlockSize, kBlockSize);
  const unsigned int tile_count = 1 + (n - 1) / kBlockSize;
  const dim3 grid_size(tile_count, tile_count);
  block_gemm_kernel<<<grid_size, block_size>>>(device_a.get(), device_b.get(), device_c.get(), n);
  cudaMemcpy(c.data(), device_c.get(), byte_count, cudaMemcpyDeviceToHost);
  return c;
}