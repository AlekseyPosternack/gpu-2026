#include "gelu_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <thread>

namespace {

constexpr int kBlockSize = 256;
constexpr int kStreamCount = 4;
constexpr size_t kChunkSize = 1 << 22;  // elements per pipeline chunk

__device__ __forceinline__ float Gelu(float x) {
    // 0.5 * x * (1 + tanh(z)) == x / (1 + exp(-2z)),
    // where z = sqrt(2 / pi) * (x + 0.044715 * x^3)
    constexpr float c1 = -1.5957691216057308f;  // -2 * sqrt(2 / pi)
    constexpr float c2 = c1 * 0.044715f;
    const float t = fminf(x * (c1 + c2 * x * x), 80.0f);
    return __fdividef(x, 1.0f + __expf(t));
}

__global__ void GeluKernel(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    const size_t n4 = n / 4;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    const float4* in4 = reinterpret_cast<const float4*>(in);
    float4* out4 = reinterpret_cast<float4*>(out);
    for (size_t i = idx; i < n4; i += stride) {
        float4 v = in4[i];
        v.x = Gelu(v.x);
        v.y = Gelu(v.y);
        v.z = Gelu(v.z);
        v.w = Gelu(v.w);
        out4[i] = v;
    }
    for (size_t i = n4 * 4 + idx; i < n; i += stride) {
        out[i] = Gelu(in[i]);
    }
}

struct Context {
    float* in = nullptr;
    float* out = nullptr;
    size_t capacity = 0;
    cudaStream_t streams[kStreamCount] = {};
    int sm_count = 0;

    Context() {
        for (auto& s : streams) {
            cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        }
        int device = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    }

    void Reserve(size_t n) {
        if (n <= capacity) {
            return;
        }
        cudaFree(in);
        cudaFree(out);
        cudaMalloc(&in, n * sizeof(float));
        cudaMalloc(&out, n * sizeof(float));
        capacity = n;
    }
};

Context& GetContext() {
    static Context ctx;
    return ctx;
}

}  // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const size_t n = input.size();
    if (n == 0) {
        return {};
    }

    Context& ctx = GetContext();
    ctx.Reserve(n);

    // allocate result on the host while the device is busy
    std::vector<float> output;
    std::thread allocator([&output, n] { output = std::vector<float>(n); });

    const size_t chunk_count = (n + kChunkSize - 1) / kChunkSize;
    for (size_t c = 0; c < chunk_count; ++c) {
        const size_t offset = c * kChunkSize;
        const size_t count = std::min(kChunkSize, n - offset);
        cudaStream_t stream = ctx.streams[c % kStreamCount];
        cudaMemcpyAsync(ctx.in + offset, input.data() + offset, count * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
        const size_t blocks = std::min<size_t>((count / 4 + kBlockSize - 1) / kBlockSize + 1,
                                               static_cast<size_t>(ctx.sm_count) * 8);
        GeluKernel<<<static_cast<unsigned>(blocks), kBlockSize, 0, stream>>>(
            ctx.in + offset, ctx.out + offset, count);
    }

    allocator.join();

    for (size_t c = 0; c < chunk_count; ++c) {
        const size_t offset = c * kChunkSize;
        const size_t count = std::min(kChunkSize, n - offset);
        cudaStream_t stream = ctx.streams[c % kStreamCount];
        cudaMemcpyAsync(output.data() + offset, ctx.out + offset, count * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
    }
    for (auto& s : ctx.streams) {
        cudaStreamSynchronize(s);
    }

    return output;
}
