#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <cstddef>

namespace {

__device__ __forceinline__ float gelu_fast(float x) {
  constexpr float kMinusTwoSqrt2OverPi = -1.5957691216057307f;
  constexpr float kCubicCoeff = kMinusTwoSqrt2OverPi * 0.044715f;
  const float arg = x * (kMinusTwoSqrt2OverPi + kCubicCoeff * x * x);
  return x / (1.0f + __expf(arg));
}

__global__ void gelu_kernel_vec4(float* data, std::size_t n) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t vec_n = n / 4;

  if (idx < vec_n) {
    float4* vec_data = reinterpret_cast<float4*>(data);
    float4 v = vec_data[idx];
    v.x = gelu_fast(v.x);
    v.y = gelu_fast(v.y);
    v.z = gelu_fast(v.z);
    v.w = gelu_fast(v.w);
    vec_data[idx] = v;
  }

  if (idx == 0) {
    for (std::size_t i = vec_n * 4; i < n; ++i) {
      data[i] = gelu_fast(data[i]);
    }
  }
}

struct CudaWorkspace {
  float* d_data = nullptr;
  std::size_t capacity = 0;
  cudaStream_t stream = nullptr;

  CudaWorkspace() {
    cudaStreamCreate(&stream);
  }

  ~CudaWorkspace() {
    if (d_data != nullptr) {
      cudaFree(d_data);
    }
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
  }

  void ensure_capacity(std::size_t size) {
    if (size > capacity) {
      if (d_data != nullptr) {
        cudaFree(d_data);
      }
      cudaMalloc(&d_data, size * sizeof(float));
      capacity = size;
    }
  }
};

} // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
  const std::size_t size = input.size();
  if (size == 0) {
    return {};
  }

  static CudaWorkspace ws;
  ws.ensure_capacity(size);

  const std::size_t bytes = size * sizeof(float);

  cudaMemcpyAsync(ws.d_data, input.data(), bytes, cudaMemcpyHostToDevice, ws.stream);

  constexpr int kBlockSize = 256;
  const std::size_t vec_n = size / 4;
  const std::size_t work_items = (vec_n > 0) ? vec_n : 1;
  const int grid_size = static_cast<int>((work_items + kBlockSize - 1) / kBlockSize);

  gelu_kernel_vec4<<<grid_size, kBlockSize, 0, ws.stream>>>(ws.d_data, size);

  std::vector<float> output(size);

  cudaMemcpyAsync(output.data(), ws.d_data, bytes, cudaMemcpyDeviceToHost, ws.stream);
  cudaStreamSynchronize(ws.stream);

  return output;
}
