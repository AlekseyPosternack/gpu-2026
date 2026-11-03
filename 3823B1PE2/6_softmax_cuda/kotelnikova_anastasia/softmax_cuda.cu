#include "softmax_cuda.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstring>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxCachedCols = 16384;

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float block_reduce_max(float v, float* shared) {
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    v = warp_reduce_max(v);
    if (lane == 0) shared[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (lane < (blockDim.x >> 5)) ? shared[lane] : -INFINITY;
        v = warp_reduce_max(v);
        if (lane == 0) shared[0] = v;
    }
    __syncthreads();
    return shared[0];
}

__device__ __forceinline__ float block_reduce_sum(float v, float* shared) {
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) shared[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (lane < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) shared[0] = v;
    }
    __syncthreads();
    return shared[0];
}

__global__ void softmax_vec4(const float* __restrict__ in,
                             float* __restrict__ out,
                             int row_size)
{
    __shared__ float shared[8];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int n4  = row_size >> 2;

    const float4* in4  = reinterpret_cast<const float4*>(in  + (size_t)row * row_size);
    float4*       out4 = reinterpret_cast<float4*>(out + (size_t)row * row_size);

    float local_max = -INFINITY;
    for (int i = tid; i < n4; i += blockDim.x) {
        const float4 v = in4[i];
        local_max = fmaxf(local_max, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
    }
    const float row_max = block_reduce_max(local_max, shared);

    float local_sum = 0.0f;
    for (int i = tid; i < n4; i += blockDim.x) {
        const float4 v = in4[i];
        float4 e;
        e.x = __expf(v.x - row_max);
        e.y = __expf(v.y - row_max);
        e.z = __expf(v.z - row_max);
        e.w = __expf(v.w - row_max);
        out4[i] = e;
        local_sum += e.x + e.y + e.z + e.w;
    }
    const float row_sum = block_reduce_sum(local_sum, shared);
    const float inv = 1.0f / row_sum;

    for (int i = tid; i < n4; i += blockDim.x) {
        float4 e = out4[i];
        e.x *= inv;
        e.y *= inv;
        e.z *= inv;
        e.w *= inv;
        out4[i] = e;
    }
}

__global__ void softmax_scalar(const float* __restrict__ in,
                               float* __restrict__ out,
                               int row_size)
{
    __shared__ float shared[8];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* src = in  + (size_t)row * row_size;
    float*       dst = out + (size_t)row * row_size;

    float local_max = -INFINITY;
    for (int i = tid; i < row_size; i += blockDim.x)
        local_max = fmaxf(local_max, src[i]);
    const float row_max = block_reduce_max(local_max, shared);

    float local_sum = 0.0f;
    for (int i = tid; i < row_size; i += blockDim.x) {
        const float e = __expf(src[i] - row_max);
        dst[i] = e;
        local_sum += e;
    }
    const float row_sum = block_reduce_sum(local_sum, shared);
    const float inv = 1.0f / row_sum;

    for (int i = tid; i < row_size; i += blockDim.x)
        dst[i] *= inv;
}

struct Workspace {
    float* h_in  = nullptr;
    float* h_out = nullptr;
    float* d_in  = nullptr;
    float* d_out = nullptr;
    size_t cap   = 0;
    cudaStream_t stream = nullptr;

    void ensure(size_t bytes) {
        if (stream == nullptr)
            cudaStreamCreate(&stream);

        if (bytes <= cap) return;

        cudaStreamSynchronize(stream);

        if (h_in)  { cudaFreeHost(h_in);  cudaFreeHost(h_out); }
        if (d_in)  { cudaFree(d_in);      cudaFree(d_out);     }

        cudaMallocHost(&h_in,  bytes);
        cudaMallocHost(&h_out, bytes);
        cudaMalloc(&d_in,  bytes);
        cudaMalloc(&d_out, bytes);

        cap = bytes;
    }
};

Workspace& workspace() {
    static Workspace ws;
    return ws;
}

} // namespace

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count)
{
    if (row_count <= 0 || input.empty())
        return {};

    const int row_size = (int)(input.size() / row_count);
    if (row_size <= 0)
        return {};

    const size_t total = input.size();
    const size_t bytes = total * sizeof(float);

    auto& ws = workspace();
    ws.ensure(bytes);

    std::memcpy(ws.h_in, input.data(), bytes);
    cudaMemcpyAsync(ws.d_in, ws.h_in, bytes, cudaMemcpyHostToDevice, ws.stream);

    if (row_size % 4 == 0)
        softmax_vec4<<<row_count, kThreads, 0, ws.stream>>>(ws.d_in, ws.d_out, row_size);
    else
        softmax_scalar<<<row_count, kThreads, 0, ws.stream>>>(ws.d_in, ws.d_out, row_size);

    cudaMemcpyAsync(ws.h_out, ws.d_out, bytes, cudaMemcpyDeviceToHost, ws.stream);
    cudaStreamSynchronize(ws.stream);

    return std::vector<float>(ws.h_out, ws.h_out + total);
}