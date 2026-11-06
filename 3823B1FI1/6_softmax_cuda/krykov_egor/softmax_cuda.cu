#include "softmax_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thread>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                        \
            std::fprintf(stderr, "CUDA error \"%s\" at %s:%d\n",          \
                         cudaGetErrorString(err_), __FILE__, __LINE__);   \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

namespace {

constexpr int kMaxThreads = 256;
constexpr size_t kMaxSharedBytes = 48 * 1024;

__device__ float BlockReduceMax(float v) {
    __shared__ float scratch[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    }
    if (lane == 0) scratch[warp] = v;
    __syncthreads();

    v = (lane < (blockDim.x >> 5)) ? scratch[lane] : -INFINITY;
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    }
    return v;
}

__device__ float BlockReduceSum(float v) {
    __shared__ float scratch[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    if (lane == 0) scratch[warp] = v;
    __syncthreads();

    v = (lane < (blockDim.x >> 5)) ? scratch[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

__global__ void SoftmaxKernel(const float* __restrict__ in,
                              float* __restrict__ out,
                              int row_size,
                              bool cached) {
    extern __shared__ float row_cache[];

    const float* x = in + static_cast<size_t>(blockIdx.x) * row_size;
    float* y = out + static_cast<size_t>(blockIdx.x) * row_size;

    float m = -INFINITY;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        const float v = x[i];
        if (cached) row_cache[i] = v;
        m = fmaxf(m, v);
    }
    m = BlockReduceMax(m);

    float sum = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        const float v = cached ? row_cache[i] : x[i];
        const float e = expf(v - m);
        if (cached) row_cache[i] = e;
        sum += e;
    }
    sum = BlockReduceSum(sum);

    const float inv = 1.0f / sum;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        const float e = cached ? row_cache[i] : expf(x[i] - m);
        y[i] = e * inv;
    }
}

struct DeviceBuffer {
    float* ptr = nullptr;
    size_t capacity = 0;

    float* Get(size_t count) {
        if (count > capacity) {
            if (ptr != nullptr) {
                CUDA_CHECK(cudaFree(ptr));
            }
            CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(float)));
            capacity = count;
        }
        return ptr;
    }

    ~DeviceBuffer() {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }
};

DeviceBuffer g_in, g_out;

}  // namespace

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (input.empty() || row_count <= 0) {
        return {};
    }
    const size_t count = input.size();
    const size_t bytes = count * sizeof(float);
    const int row_size = static_cast<int>(count / row_count);

    float* d_in = g_in.Get(count);
    float* d_out = g_out.Get(count);

    std::vector<float> result;
    std::thread alloc_thread([&result, count] { result.resize(count); });

    CUDA_CHECK(cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice));

    const int block = std::min(kMaxThreads, std::max(32, row_size));
    const size_t shared_bytes = static_cast<size_t>(row_size) * sizeof(float);
    const bool cached = shared_bytes <= kMaxSharedBytes;

    SoftmaxKernel<<<row_count, block, cached ? shared_bytes : 0>>>(d_in, d_out, row_size, cached);
    CUDA_CHECK(cudaGetLastError());

    alloc_thread.join();

    CUDA_CHECK(cudaMemcpy(result.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    return result;
}