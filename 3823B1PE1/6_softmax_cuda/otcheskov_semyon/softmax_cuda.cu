#include "softmax_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {


constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kThreads / kWarpSize;
constexpr int kLaneMask = kWarpSize - 1;
constexpr int kWarpShift = 5;
constexpr unsigned int kFullWarpMask = 0xffffffffu;
constexpr int kShuffleStart = kWarpSize / 2;
constexpr int kVecWidth = 4;

constexpr float kNegInf = -std::numeric_limits<float>::infinity();
constexpr float kZero = 0.0f;
constexpr float kOne = 1.0f;

} // namespace


#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        if (cudaError_t s = (call); s != cudaSuccess) {                      \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,      \
                         __LINE__, cudaGetErrorString(s));                   \
            throw std::runtime_error("CUDA error");                          \
        }                                                                    \
    } while (0)

namespace {



__device__ float4 load_float4(const float* p) {
    return *reinterpret_cast<const float4*>(p);
}

__device__ void store_float4(float* p, const float4& v) {
    *reinterpret_cast<float4*>(p) = v;
}

__device__ float warpMax(float v) {
    for (int offset = kShuffleStart; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(kFullWarpMask, v, offset));
    }
    return v;
}

__device__ float warpSum(float v) {
    for (int offset = kShuffleStart; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(kFullWarpMask, v, offset);
    }
    return v;
}

__device__ float blockMax(float v, float* shared) {
    const int lane = threadIdx.x & kLaneMask;
    const int warp = threadIdx.x >> kWarpShift;

    v = warpMax(v);
    if (lane == 0)
      shared[warp] = v;

    __syncthreads();

    v = (threadIdx.x < kWarpsPerBlock) ? shared[threadIdx.x] : kNegInf;
    if (warp == 0)
      v = warpMax(v);

    if (threadIdx.x == 0)
      shared[0] = v;

    __syncthreads();

    return shared[0];
}

__device__ float blockSum(float v, float* shared) {
    const int lane = threadIdx.x & kLaneMask;
    const int warp = threadIdx.x >> kWarpShift;

    v = warpSum(v);
    if (lane == 0)
      shared[warp] = v;

    __syncthreads();

    v = (threadIdx.x < kWarpsPerBlock) ? shared[threadIdx.x] : kZero;
    if (warp == 0)
      v = warpSum(v);

    if (threadIdx.x == 0)
      shared[0] = v;

    __syncthreads();

    return shared[0];
}

__global__ void softmax_kernel(const float* input, float* output, int row_size) {
    extern __shared__ float shared_row[];
    __shared__ float partials[kWarpsPerBlock];

    const float* input_row = 
        input + static_cast<std::size_t>(blockIdx.x) * row_size;

    float* output_row = 
        output + static_cast<std::size_t>(blockIdx.x) * row_size;

    constexpr int kStride = kThreads * kVecWidth;

    for (int i = threadIdx.x * kVecWidth; i < row_size; i += kStride) {
        store_float4(shared_row + i, load_float4(input_row + i));
    }
    __syncthreads();

    float thread_max = kNegInf;
    for (int i = threadIdx.x * kVecWidth; i < row_size; i += kStride) {
        const float4 v = load_float4(shared_row + i);
        thread_max = fmaxf(thread_max, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
    }
    const float row_max = blockMax(thread_max, partials);

    float thread_sum = kZero;
    for (int i = threadIdx.x * kVecWidth; i < row_size; i += kStride) {
        float4 v = load_float4(shared_row + i);

        v.x = expf(v.x - row_max);
        v.y = expf(v.y - row_max);
        v.z = expf(v.z - row_max);
        v.w = expf(v.w - row_max);
        
        store_float4(shared_row + i, v);
        thread_sum += v.x + v.y + v.z + v.w;
    }
    const float inverse_sum = kOne / blockSum(thread_sum, partials);

    for (int i = threadIdx.x * kVecWidth; i < row_size; i += kStride) {
        const float4 v = load_float4(shared_row + i);
        const float4 norm_v = make_float4(v.x * inverse_sum,
                                          v.y * inverse_sum,
                                          v.z * inverse_sum,
                                          v.w * inverse_sum);
        store_float4(output_row + i, norm_v);
    }
}

struct CachedBuffers {
    float* device_input = nullptr;
    float* device_output = nullptr;
    float* pinned_output = nullptr;
    std::size_t capacity_bytes = 0;
    cudaStream_t stream = nullptr;
    bool kernel_attr_set = false;
    int configured_shared_bytes = 0;
};

CachedBuffers& get_buffers() {
    static CachedBuffers instance;
    return instance;
}

void ensure_capacity(CachedBuffers& b, std::size_t bytes) {
    if (bytes <= b.capacity_bytes) return;

    if (b.device_input)  CUDA_CHECK(cudaFree(b.device_input));
    if (b.device_output) CUDA_CHECK(cudaFree(b.device_output));
    if (b.pinned_output) CUDA_CHECK(cudaFreeHost(b.pinned_output));

    CUDA_CHECK(cudaMalloc(&b.device_input, bytes));
    CUDA_CHECK(cudaMalloc(&b.device_output, bytes));
    CUDA_CHECK(cudaMallocHost(&b.pinned_output, bytes));

    b.capacity_bytes = bytes;
}

} // namespace


std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    const std::size_t element_count = input.size();
    const int row_size = static_cast<int>(element_count / row_count);
    const std::size_t buffer_bytes = element_count * sizeof(float);

    CachedBuffers& b = get_buffers();
    if (!b.stream) {
        CUDA_CHECK(cudaStreamCreate(&b.stream));
    }
    ensure_capacity(b, buffer_bytes);

    const std::size_t shared_bytes = static_cast<std::size_t>(row_size) * sizeof(float);
    const int shared_bytes_int = static_cast<int>(shared_bytes);

    if (!b.kernel_attr_set || b.configured_shared_bytes < shared_bytes_int) {
        CUDA_CHECK(cudaFuncSetAttribute(
            softmax_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_bytes_int));
        b.kernel_attr_set = true;
        b.configured_shared_bytes = shared_bytes_int;
    }

    CUDA_CHECK(cudaMemcpyAsync(b.device_input, input.data(), buffer_bytes,
                               cudaMemcpyHostToDevice, b.stream));

    softmax_kernel<<<row_count, kThreads, shared_bytes, b.stream>>>(
        b.device_input, b.device_output, row_size);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(b.pinned_output, b.device_output, buffer_bytes,
                               cudaMemcpyDeviceToHost, b.stream));
    
    std::vector<float> output(element_count);
    CUDA_CHECK(cudaStreamSynchronize(b.stream));

    std::memcpy(output.data(), b.pinned_output, buffer_bytes);
    return output;
}