#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>
#include <cstddef>

namespace {

__global__ void naive_gemm_small_kernel(const float* __restrict__ a,
                                        const float* __restrict__ b,
                                        float* __restrict__ c,
                                        int n) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < n && col < n) {
    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
      sum += a[row * n + k] * b[k * n + col];
    }
    c[row * n + col] = sum;
  }
}

__global__ void naive_gemm_vec4_kernel(const float* __restrict__ a,
                                       const float* __restrict__ b,
                                       float* __restrict__ c,
                                       int n) {
  const int col4 = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
  const int n4 = n / 4;

  if (row >= n || col4 >= n4) {
    return;
  }

  const float4* __restrict__ b4 = reinterpret_cast<const float4*>(b);
  float4* __restrict__ c4 = reinterpret_cast<float4*>(c);

  float4 c0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float4 c1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float4 c2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float4 c3 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

  const float* a_row0 = a + (row + 0) * n;
  const float* a_row1 = a + (row + 1) * n;
  const float* a_row2 = a + (row + 2) * n;
  const float* a_row3 = a + (row + 3) * n;

  #pragma unroll 4
  for (int k = 0; k < n; ++k) {
    const float4 bv = b4[k * n4 + col4];

    const float a0 = a_row0[k];
    const float a1 = a_row1[k];
    const float a2 = a_row2[k];
    const float a3 = a_row3[k];

    c0.x = fmaf(a0, bv.x, c0.x);
    c0.y = fmaf(a0, bv.y, c0.y);
    c0.z = fmaf(a0, bv.z, c0.z);
    c0.w = fmaf(a0, bv.w, c0.w);

    c1.x = fmaf(a1, bv.x, c1.x);
    c1.y = fmaf(a1, bv.y, c1.y);
    c1.z = fmaf(a1, bv.z, c1.z);
    c1.w = fmaf(a1, bv.w, c1.w);

    c2.x = fmaf(a2, bv.x, c2.x);
    c2.y = fmaf(a2, bv.y, c2.y);
    c2.z = fmaf(a2, bv.z, c2.z);
    c2.w = fmaf(a2, bv.w, c2.w);

    c3.x = fmaf(a3, bv.x, c3.x);
    c3.y = fmaf(a3, bv.y, c3.y);
    c3.z = fmaf(a3, bv.z, c3.z);
    c3.w = fmaf(a3, bv.w, c3.w);
  }

  c4[(row + 0) * n4 + col4] = c0;
  c4[(row + 1) * n4 + col4] = c1;
  c4[(row + 2) * n4 + col4] = c2;
  c4[(row + 3) * n4 + col4] = c3;
}

struct GemmWorkspace {
  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;
  std::size_t capacity = 0;
  cudaStream_t stream = nullptr;

  GemmWorkspace() {
    cudaStreamCreate(&stream);
  }

  ~GemmWorkspace() {
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

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
  if (n <= 0 || a.empty() || b.empty()) {
    return {};
  }

  const std::size_t total_elems = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
  const std::size_t bytes = total_elems * sizeof(float);

  static GemmWorkspace ws;
  ws.ensure_capacity(total_elems);

  cudaMemcpyAsync(ws.d_a, a.data(), bytes, cudaMemcpyHostToDevice, ws.stream);
  cudaMemcpyAsync(ws.d_b, b.data(), bytes, cudaMemcpyHostToDevice, ws.stream);

  if (n < 4) {
    dim3 block(n, n);
    dim3 grid(1, 1);
    naive_gemm_small_kernel<<<grid, block, 0, ws.stream>>>(ws.d_a, ws.d_b, ws.d_c, n);
  } else {
    const dim3 block(32, 8);
    const int n4 = n / 4;
    const dim3 grid((n4 + block.x - 1) / block.x,
                    (n4 + block.y - 1) / block.y);
    naive_gemm_vec4_kernel<<<grid, block, 0, ws.stream>>>(ws.d_a, ws.d_b, ws.d_c, n);
  }

  std::vector<float> result(total_elems);

  cudaMemcpyAsync(result.data(), ws.d_c, bytes, cudaMemcpyDeviceToHost, ws.stream);
  cudaStreamSynchronize(ws.stream);

  return result;
}
