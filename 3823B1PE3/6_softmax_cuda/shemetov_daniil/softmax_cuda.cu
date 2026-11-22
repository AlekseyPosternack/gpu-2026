#include "softmax_cuda.h"

#include <cuda_runtime.h>
#include <cfloat>
#include <cstddef>

namespace {

constexpr int kMaxBlockSize = 256;

__device__ __forceinline__ void online_softmax_update(float val,
                                                      float& max_val,
                                                      float& sum_exp) {
  if (val > max_val) {
    sum_exp = sum_exp * __expf(max_val - val) + 1.0f;
    max_val = val;
  } else {
    sum_exp += __expf(val - max_val);
  }
}

__global__ void softmax_small_kernel(float* data, int row_count, int row_size) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= row_count) {
    return;
  }

  float* row_ptr = data + static_cast<std::size_t>(row) * row_size;

  float row_max = -FLT_MAX;
  for (int i = 0; i < row_size; ++i) {
    row_max = fmaxf(row_max, row_ptr[i]);
  }

  float sum = 0.0f;
  for (int i = 0; i < row_size; ++i) {
    const float e = __expf(row_ptr[i] - row_max);
    row_ptr[i] = e;
    sum += e;
  }

  const float inv_sum = 1.0f / sum;
  for (int i = 0; i < row_size; ++i) {
    row_ptr[i] *= inv_sum;
  }
}

__global__ void softmax_vec4_kernel(float* data, int vec_row_size) {
  __shared__ float s_buf[kMaxBlockSize];

  const int tid = threadIdx.x;
  const int bdim = blockDim.x;
  const std::size_t row = static_cast<std::size_t>(blockIdx.x);
  float4* row_ptr = reinterpret_cast<float4*>(data) + row * vec_row_size;

  float local_max = -FLT_MAX;
  float local_sum = 0.0f;

  for (int i = tid; i < vec_row_size; i += bdim) {
    const float4 v = row_ptr[i];
    online_softmax_update(v.x, local_max, local_sum);
    online_softmax_update(v.y, local_max, local_sum);
    online_softmax_update(v.z, local_max, local_sum);
    online_softmax_update(v.w, local_max, local_sum);
  }

  s_buf[tid] = local_max;
  __syncthreads();

  for (int stride = bdim >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_buf[tid] = fmaxf(s_buf[tid], s_buf[tid + stride]);
    }
    __syncthreads();
  }
  const float row_max = s_buf[0];
  __syncthreads();

  s_buf[tid] = (local_sum > 0.0f) ? (local_sum * __expf(local_max - row_max)) : 0.0f;
  __syncthreads();

  for (int stride = bdim >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_buf[tid] += s_buf[tid + stride];
    }
    __syncthreads();
  }
  const float inv_sum = 1.0f / s_buf[0];

  for (int i = tid; i < vec_row_size; i += bdim) {
    float4 v = row_ptr[i];
    v.x = __expf(v.x - row_max) * inv_sum;
    v.y = __expf(v.y - row_max) * inv_sum;
    v.z = __expf(v.z - row_max) * inv_sum;
    v.w = __expf(v.w - row_max) * inv_sum;
    row_ptr[i] = v;
  }
}

struct SoftmaxWorkspace {
  float* d_data = nullptr;
  std::size_t capacity = 0;
  cudaStream_t stream = nullptr;

  SoftmaxWorkspace() {
    cudaStreamCreate(&stream);
  }

  ~SoftmaxWorkspace() {
    if (d_data != nullptr) cudaFree(d_data);
    if (stream != nullptr) cudaStreamDestroy(stream);
  }

  void ensure_capacity(std::size_t total_elems) {
    if (total_elems > capacity) {
      if (d_data != nullptr) cudaFree(d_data);
      cudaMalloc(&d_data, total_elems * sizeof(float));
      capacity = total_elems;
    }
  }
};

} // namespace

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
  const std::size_t total_elems = input.size();
  if (total_elems == 0 || row_count <= 0) {
    return {};
  }

  const int row_size = static_cast<int>(total_elems / static_cast<std::size_t>(row_count));
  const std::size_t bytes = total_elems * sizeof(float);

  static SoftmaxWorkspace ws;
  ws.ensure_capacity(total_elems);

  cudaMemcpyAsync(ws.d_data, input.data(), bytes, cudaMemcpyHostToDevice, ws.stream);

  if (row_size < 4) {
    constexpr int kSmallBlock = 256;
    const int grid = (row_count + kSmallBlock - 1) / kSmallBlock;
    softmax_small_kernel<<<grid, kSmallBlock, 0, ws.stream>>>(ws.d_data, row_count, row_size);
  } else {
    const int vec_row_size = row_size / 4;
    int block_size = kMaxBlockSize;
    while (block_size > vec_row_size && block_size > 32) {
      block_size >>= 1;
    }
    softmax_vec4_kernel<<<row_count, block_size, 0, ws.stream>>>(ws.d_data, vec_row_size);
  }

  std::vector<float> result(total_elems);

  cudaMemcpyAsync(result.data(), ws.d_data, bytes, cudaMemcpyDeviceToHost, ws.stream);
  cudaStreamSynchronize(ws.stream);

  return result;
}
