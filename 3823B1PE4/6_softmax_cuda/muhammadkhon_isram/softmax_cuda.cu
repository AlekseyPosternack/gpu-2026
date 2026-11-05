#include "softmax_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>

namespace {

constexpr int kMaxBlock = 512;
constexpr int kStreams = 4;
constexpr size_t kChunkElems = 1u << 20;
constexpr size_t kMaxCachedRow = 8192;

struct MaxOp {
    __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
};
struct SumOp {
    __device__ float operator()(float a, float b) const { return a + b; }
};

template <typename Op>
__device__ __forceinline__ float BlockReduce(float v, float* red, float identity) {
    Op op;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = op(v, __shfl_xor_sync(0xffffffff, v, o));

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();

    v = lane < (blockDim.x >> 5) ? red[lane] : identity;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = op(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

template <bool kCache>
__global__ void SoftmaxKernel(const float* __restrict__ in,
                              float* __restrict__ out, int row_size) {
    extern __shared__ float cache[];
    __shared__ float red[32];

    const size_t base = static_cast<size_t>(blockIdx.x) * row_size;
    const float* row_in = in + base;
    float* row_out = out + base;

    float m = -INFINITY;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        const float x = row_in[i];
        if (kCache) cache[i] = x;
        m = fmaxf(m, x);
    }
    m = BlockReduce<MaxOp>(m, red, -INFINITY);

    float s = 0.f;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        const float x = kCache ? cache[i] : row_in[i];
        const float e = __expf(x - m);
        if (kCache) cache[i] = e;
        s += e;
    }
    s = BlockReduce<SumOp>(s, red, 0.f);
    const float inv = 1.0f / s;

    // 3) нормировка
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        const float e = kCache ? cache[i] : __expf(row_in[i] - m);
        row_out[i] = e * inv;
    }
}

struct Context {
    cudaStream_t streams[kStreams];
    float* d_in = nullptr;
    float* d_out = nullptr;
    size_t capacity = 0;

    Context() {
        for (auto& s : streams) cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    }
    ~Context() {
        for (auto& s : streams) cudaStreamDestroy(s);
        cudaFree(d_in);
        cudaFree(d_out);
    }
    void Reserve(size_t n) {
        if (n <= capacity) return;
        cudaFree(d_in);
        cudaFree(d_out);
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        capacity = n;
    }
};

}  // namespace

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    const size_t n = input.size();
    if (n == 0 || row_count <= 0) return std::vector<float>(n);

    const size_t row_size = n / row_count;

    static Context ctx;
    ctx.Reserve(n);

    int block = 32;
    while (block < kMaxBlock && static_cast<size_t>(block) < row_size) block *= 2;

    const bool cached = row_size <= kMaxCachedRow;
    const size_t smem = cached ? row_size * sizeof(float) : 0;

    const size_t rows_per_chunk =
        std::max<size_t>(1, kChunkElems / row_size);
    const size_t chunks = (row_count + rows_per_chunk - 1) / rows_per_chunk;

    for (size_t c = 0; c < chunks; ++c) {
        const size_t r0 = c * rows_per_chunk;
        const size_t rows = std::min(rows_per_chunk, row_count - r0);
        const size_t off = r0 * row_size;
        cudaStream_t s = ctx.streams[c % kStreams];

        cudaMemcpyAsync(ctx.d_in + off, input.data() + off,
                        rows * row_size * sizeof(float), cudaMemcpyHostToDevice, s);
        if (cached)
            SoftmaxKernel<true><<<static_cast<unsigned>(rows), block, smem, s>>>(
                ctx.d_in + off, ctx.d_out + off, static_cast<int>(row_size));
        else
            SoftmaxKernel<false><<<static_cast<unsigned>(rows), block, 0, s>>>(
                ctx.d_in + off, ctx.d_out + off, static_cast<int>(row_size));
    }

    std::vector<float> output(n);

    for (size_t c = 0; c < chunks; ++c) {
        const size_t r0 = c * rows_per_chunk;
        const size_t rows = std::min(rows_per_chunk, row_count - r0);
        const size_t off = r0 * row_size;
        cudaMemcpyAsync(output.data() + off, ctx.d_out + off,
                        rows * row_size * sizeof(float), cudaMemcpyDeviceToHost,
                        ctx.streams[c % kStreams]);
    }
    for (auto& s : ctx.streams) cudaStreamSynchronize(s);

    return output;
}