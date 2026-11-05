#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>

namespace {

constexpr int kThreads = 256;
constexpr int kStreams = 4;
constexpr std::size_t kChunk = 1u << 20;

constexpr float kC1 = -1.5957691216057308f;
constexpr float kC2 = -1.5957691216057308f * 0.044715f;

__device__ __forceinline__ float Gelu(float x) {
    const float t = x * (kC1 + kC2 * x * x);
    return x / (1.0f + expf(t));
}

__global__ void GeluKernel(float* data, std::size_t n) {
    const std::size_t n4 = n / 4;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    float4* data4 = reinterpret_cast<float4*>(data);
    for (; i < n4; i += stride) {
        float4 v = data4[i];
        v.x = Gelu(v.x);
        v.y = Gelu(v.y);
        v.z = Gelu(v.z);
        v.w = Gelu(v.w);
        data4[i] = v;
    }

    const std::size_t tail = n - n4 * 4;
    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < tail) {
        const std::size_t k = n4 * 4 + tid;
        data[k] = Gelu(data[k]);
    }
}

struct Context {
    cudaStream_t streams[kStreams];
    float* d_buf = nullptr;
    std::size_t capacity = 0;

    Context() {
        for (int i = 0; i < kStreams; ++i)
            cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }
    ~Context() {
        for (int i = 0; i < kStreams; ++i) cudaStreamDestroy(streams[i]);
        if (d_buf) cudaFree(d_buf);
    }
    void Reserve(std::size_t n) {
        if (n <= capacity) return;
        if (d_buf) cudaFree(d_buf);
        cudaMalloc(&d_buf, n * sizeof(float));
        capacity = n;
    }
};

}  // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const std::size_t n = input.size();
    if (n == 0) return {};

    static Context ctx;
    ctx.Reserve(n);

    const float* h_in = input.data();
    const std::size_t chunks = (n + kChunk - 1) / kChunk;

    for (std::size_t c = 0; c < chunks; ++c) {
        const std::size_t off = c * kChunk;
        const std::size_t len = std::min(kChunk, n - off);
        cudaStream_t s = ctx.streams[c % kStreams];

        cudaMemcpyAsync(ctx.d_buf + off, h_in + off, len * sizeof(float),
                        cudaMemcpyHostToDevice, s);

        const std::size_t work = (len + 3) / 4;
        const int blocks = static_cast<int>(
            std::min<std::size_t>((work + kThreads - 1) / kThreads, 65535));
        GeluKernel<<<blocks, kThreads, 0, s>>>(ctx.d_buf + off, len);
    }

    std::vector<float> result(n);
    float* h_out = result.data();

    for (std::size_t c = 0; c < chunks; ++c) {
        const std::size_t off = c * kChunk;
        const std::size_t len = std::min(kChunk, n - off);
        cudaMemcpyAsync(h_out + off, ctx.d_buf + off, len * sizeof(float),
                        cudaMemcpyDeviceToHost, ctx.streams[c % kStreams]);
    }

    for (int i = 0; i < kStreams; ++i) cudaStreamSynchronize(ctx.streams[i]);

    return result;
}
