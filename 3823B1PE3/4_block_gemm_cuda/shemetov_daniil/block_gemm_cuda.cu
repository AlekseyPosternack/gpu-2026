#include "block_gemm_cuda.h"

#include <cuda_runtime.h>
#include <cstddef>

namespace {

constexpr int kBM = 64;
constexpr int kBN = 64;
constexpr int kBK = 16;

__global__ void block_gemm_small_kernel(const float* __restrict__ a,
                                        const float* __restrict__ b,
                                        float* __restrict__ c,
                                        int n) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < n && col < n) {
    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
      sum = fmaf(a[row * n + k], b[k * n + col], sum);
    }
    c[row * n + col] = sum;
  }
}

__global__ void block_gemm_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c,
                                  int n) {
  __shared__ float As[kBM][kBK + 1];
  __shared__ float Bs[kBK][kBN];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * 16 + tx;

  const int block_row = blockIdx.y * kBM;
  const int block_col = blockIdx.x * kBN;

  const int a_row = tid / 4;
  const int a_col = (tid % 4) * 4;

  const int b_row = tid / 16;
  const int b_col = (tid % 16) * 4;

  float acc[4][4] = {{0.0f}};

  for (int tile = 0; tile < n; tile += kBK) {
    const float4 a_vec = *reinterpret_cast<const float4*>(
        a + (block_row + a_row) * n + tile + a_col);
    As[a_row][a_col + 0] = a_vec.x;
    As[a_row][a_col + 1] = a_vec.y;
    As[a_row][a_col + 2] = a_vec.z;
    As[a_row][a_col + 3] = a_vec.w;

    const float4 b_vec = *reinterpret_cast<const float4*>(
        b + (tile + b_row) * n + block_col + b_col);
    Bs[b_row][b_col + 0] = b_vec.x;
    Bs[b_row][b_col + 1] = b_vec.y;
    Bs[b_row][b_col + 2] = b_vec.z;
    Bs[b_row][b_col + 3] = b_vec.w;

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < kBK; ++k) {
      float a_reg[4];
      float b_reg[4];

      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        a_reg[i] = As[ty + i * 16][k];
        b_reg[i] = Bs[k][tx + i * 16];
      }

      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          acc[i][j] = fmaf(a_reg[i], b_reg[j], acc[i][j]);
        }
      }
    }

    __syncthreads();
  }

  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int row = block_row + ty + i * 16;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int col = block_col + tx + j * 16;
      c[row * n + col] = acc[i][j];
    }
  }
}

struct BlockGemmWorkspace {
  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;
  std::size_t capacity = 0;
  cudaStream_t stream = nullptr;

  BlockGemmWorkspace() {
    cudaStreamCreate(&stream);
  }

  ~BlockGemmWorkspace() {
    if (d_a != nullptr) cudaFree(d_a);
    if (d_b != nullptr) cudaFree(d_b);
    if (d_c != nullptr) cudaFree(d_c);
    if (stream != nullptr) cudaStreamDestroy(stream);
  }

  void ensure_capacity(std::size_t total_elems) {
    if (total_elems > capacity) {
      if (d_a != nullptr) cudaFree(d_a);
      if (d_b != nullptr) cudaFree(d_b);
      if (d_c != nullptr) cudaFree(d_c);

      const std::size_t bytes = total_elems * sizeof(float);
      cudaMalloc(&d_a, bytes);
      cudaMalloc(&d_b, bytes);
      cudaMalloc(&d_c, bytes);
      capacity = total_elems;
    }
  }
};

} // namespace

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
  if (n <= 0 || a.empty() || b.empty()) {
    return {};
  }

  const std::size_t total_elems = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
  const std::size_t bytes = total_elems * sizeof(float);

  static BlockGemmWorkspace ws;
  ws.ensure_capacity(total_elems);

  cudaMemcpyAsync(ws.d_a, a.data(), bytes, cudaMemcpyHostToDevice, ws.stream);
  cudaMemcpyAsync(ws.d_b, b.data(), bytes, cudaMemcpyHostToDevice, ws.stream);

  if (n < kBM) {
    const dim3 block(16, 16);
    const dim3 grid((n + block.x - 1) / block.x,
                    (n + block.y - 1) / block.y);
    block_gemm_small_kernel<<<grid, block, 0, ws.stream>>>(ws.d_a, ws.d_b, ws.d_c, n);
  } else {
    const dim3 block(16, 16);
    const dim3 grid(n / kBN, n / kBM);
    block_gemm_kernel<<<grid, block, 0, ws.stream>>>(ws.d_a, ws.d_b, ws.d_c, n);
  }

  std::vector<float> result(total_elems);

  cudaMemcpyAsync(result.data(), ws.d_c, bytes, cudaMemcpyDeviceToHost, ws.stream);
  cudaStreamSynchronize(ws.stream);

  return result;
}
