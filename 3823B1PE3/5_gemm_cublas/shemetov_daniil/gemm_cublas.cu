#include "gemm_cublas.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstddef>

namespace {

struct CublasWorkspace {
  cublasHandle_t handle = nullptr;
  cudaStream_t stream = nullptr;
  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;
  std::size_t capacity = 0;

  CublasWorkspace() {
    cudaStreamCreate(&stream);
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
  }

  ~CublasWorkspace() {
    if (d_a != nullptr) cudaFree(d_a);
    if (d_b != nullptr) cudaFree(d_b);
    if (d_c != nullptr) cudaFree(d_c);
    if (handle != nullptr) cublasDestroy(handle);
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

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
  if (n <= 0 || a.empty() || b.empty()) {
    return {};
  }

  const std::size_t total_elems = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
  const std::size_t bytes = total_elems * sizeof(float);

  static CublasWorkspace ws;
  ws.ensure_capacity(total_elems);

  cudaMemcpyAsync(ws.d_a, a.data(), bytes, cudaMemcpyHostToDevice, ws.stream);
  cudaMemcpyAsync(ws.d_b, b.data(), bytes, cudaMemcpyHostToDevice, ws.stream);

  const float alpha = 1.0f;
  const float beta = 0.0f;

  cublasSgemm(ws.handle,
              CUBLAS_OP_N, CUBLAS_OP_N,
              n, n, n,
              &alpha,
              ws.d_b, n,
              ws.d_a, n,
              &beta,
              ws.d_c, n);

  std::vector<float> result(total_elems);

  cudaMemcpyAsync(result.data(), ws.d_c, bytes, cudaMemcpyDeviceToHost, ws.stream);
  cudaStreamSynchronize(ws.stream);

  return result;
}
