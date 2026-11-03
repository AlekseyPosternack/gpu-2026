#include "softmax_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <thread>

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kMaxVecs = 16;
constexpr size_t kChunk = size_t(1) << 22;
constexpr size_t kSmall = size_t(1) << 20;
constexpr size_t kPage = 4096;
constexpr int kStreams = 4;

struct Max {
    __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
};

struct Sum {
    __device__ float operator()(float a, float b) const { return a + b; }
};

template <class Op>
__device__ float BlockReduce(float value, float* shared, Op op) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value = op(value, __shfl_xor_sync(0xffffffffu, value, offset));
    }
    if (threadIdx.x % 32 == 0) {
        shared[threadIdx.x / 32] = value;
    }
    __syncthreads();
    float result = shared[0];
    for (int i = 1; i < kWarps; ++i) {
        result = op(result, shared[i]);
    }
    __syncthreads();
    return result;
}

template <int kVecs>
__global__ void __launch_bounds__(kThreads) SoftmaxCached(float* data, int row_size) {
    __shared__ float shared[kWarps];
    float4* row = reinterpret_cast<float4*>(data + size_t(blockIdx.x) * row_size);
    const int vecs = row_size / 4;

    float4 v[kVecs];
    float max = -INFINITY;
#pragma unroll
    for (int i = 0; i < kVecs; ++i) {
        const int j = threadIdx.x + i * kThreads;
        v[i] = j < vecs ? row[j] : make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
        max = fmaxf(max, fmaxf(fmaxf(v[i].x, v[i].y), fmaxf(v[i].z, v[i].w)));
    }
    max = BlockReduce(max, shared, Max());

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecs; ++i) {
        v[i].x = __expf(v[i].x - max);
        v[i].y = __expf(v[i].y - max);
        v[i].z = __expf(v[i].z - max);
        v[i].w = __expf(v[i].w - max);
        sum += (v[i].x + v[i].y) + (v[i].z + v[i].w);
    }
    const float inv = 1.0f / BlockReduce(sum, shared, Sum());

#pragma unroll
    for (int i = 0; i < kVecs; ++i) {
        const int j = threadIdx.x + i * kThreads;
        if (j < vecs) {
            row[j] = make_float4(v[i].x * inv, v[i].y * inv, v[i].z * inv, v[i].w * inv);
        }
    }
}

__global__ void __launch_bounds__(kThreads) SoftmaxGeneric(float* data, int row_size) {
    __shared__ float shared[kWarps];
    float* row = data + size_t(blockIdx.x) * row_size;

    float max = -INFINITY;
    for (int j = threadIdx.x; j < row_size; j += kThreads) {
        max = fmaxf(max, row[j]);
    }
    max = BlockReduce(max, shared, Max());

    float sum = 0.0f;
    for (int j = threadIdx.x; j < row_size; j += kThreads) {
        sum += __expf(row[j] - max);
    }
    const float inv = 1.0f / BlockReduce(sum, shared, Sum());

    for (int j = threadIdx.x; j < row_size; j += kThreads) {
        row[j] = __expf(row[j] - max) * inv;
    }
}

void Launch(float* data, size_t rows, int row_size, cudaStream_t stream) {
    const unsigned grid = static_cast<unsigned>(rows);
    const int vecs = row_size / 4;
    if (row_size % 4 != 0 || vecs > kMaxVecs * kThreads) {
        SoftmaxGeneric<<<grid, kThreads, 0, stream>>>(data, row_size);
    } else if (vecs <= kThreads) {
        SoftmaxCached<1><<<grid, kThreads, 0, stream>>>(data, row_size);
    } else if (vecs <= 2 * kThreads) {
        SoftmaxCached<2><<<grid, kThreads, 0, stream>>>(data, row_size);
    } else if (vecs <= 4 * kThreads) {
        SoftmaxCached<4><<<grid, kThreads, 0, stream>>>(data, row_size);
    } else if (vecs <= 8 * kThreads) {
        SoftmaxCached<8><<<grid, kThreads, 0, stream>>>(data, row_size);
    } else {
        SoftmaxCached<16><<<grid, kThreads, 0, stream>>>(data, row_size);
    }
}

struct Context {
    float* device = nullptr;
    size_t capacity = 0;
    cudaStream_t streams[kStreams];
    cudaEvent_t idle[kStreams];
    std::vector<cudaEvent_t> computed;
    std::vector<cudaEvent_t> received;

    Context() {
        for (int i = 0; i < kStreams; ++i) {
            cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
            cudaEventCreateWithFlags(&idle[i], cudaEventDisableTiming);
        }
    }

    ~Context() {
        cudaFree(device);
        for (int i = 0; i < kStreams; ++i) {
            cudaStreamDestroy(streams[i]);
            cudaEventDestroy(idle[i]);
        }
        for (cudaEvent_t e : computed) {
            cudaEventDestroy(e);
        }
        for (cudaEvent_t e : received) {
            cudaEventDestroy(e);
        }
    }

    void Reserve(size_t n, size_t chunks) {
        if (n > capacity) {
            cudaFree(device);
            cudaMalloc(&device, n * sizeof(float));
            capacity = n;
        }
        for (std::vector<cudaEvent_t>* events : {&computed, &received}) {
            while (events->size() < chunks) {
                cudaEvent_t e;
                cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
                events->push_back(e);
            }
        }
    }
};

Context& GetContext() {
    static Context context;
    return context;
}

void WaitFor(const std::atomic<size_t>& counter, size_t value) {
    while (counter.load(std::memory_order_acquire) <= value) {
        std::this_thread::yield();
    }
}

}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    const size_t n = input.size();
    const int row_size = row_count > 0 ? static_cast<int>(n / row_count) : 0;
    const size_t rows_per_part = std::max<size_t>(1, kChunk / std::max(row_size, 1));
    const size_t part = rows_per_part * row_size;
    const size_t parts = part > 0 ? (n + part - 1) / part : 0;

    Context& ctx = GetContext();
    ctx.Reserve(n, parts + 1);

    if (n < kSmall) {
        std::vector<float> output(n);
        if (n > 0) {
            cudaMemcpy(ctx.device, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);
            Launch(ctx.device, row_count, row_size, 0);
            cudaMemcpy(output.data(), ctx.device, n * sizeof(float), cudaMemcpyDeviceToHost);
        }
        return output;
    }

    std::vector<float> output;
    output.reserve(n);
    float* out = output.data();
    const size_t shift = reinterpret_cast<uintptr_t>(out) % kPage / sizeof(float);
    const size_t chunks = (n + shift + part - 1) / part;
    auto begin = [&](size_t c) { return c == 0 ? 0 : c * part - shift; };
    auto end = [&](size_t c) { return std::min(n, (c + 1) * part - shift); };

    std::atomic<size_t> built{0};
    std::atomic<size_t> issued{0};
    std::atomic<size_t> received{0};

    std::thread builder([&] {
        for (size_t c = 0; c < chunks; ++c) {
            output.resize(end(c));
            built.store(c + 1, std::memory_order_release);
        }
    });

    std::thread receiver([&] {
        for (size_t c = 0; c < chunks; ++c) {
            const size_t bytes = (end(c) - begin(c)) * sizeof(float);
            const size_t first = begin(c) / part;
            const size_t last = (end(c) - 1) / part;
            cudaStream_t stream = ctx.streams[c % kStreams];
            WaitFor(built, c);
            cudaHostRegister(out + begin(c), bytes, cudaHostRegisterDefault);
            WaitFor(issued, last);
            for (size_t p = first; p <= last; ++p) {
                cudaStreamWaitEvent(stream, ctx.computed[p], 0);
            }
            cudaMemcpyAsync(out + begin(c), ctx.device + begin(c), bytes, cudaMemcpyDeviceToHost, stream);
            cudaEventRecord(ctx.received[c], stream);
            received.store(c + 1, std::memory_order_release);
        }
    });

    float* in = const_cast<float*>(input.data());
    cudaHostRegister(in, n * sizeof(float), cudaHostRegisterDefault);
    for (size_t p = 0; p < parts; ++p) {
        const size_t len = std::min(part, n - p * part);
        cudaStream_t stream = ctx.streams[p % kStreams];
        cudaMemcpyAsync(ctx.device + p * part, in + p * part, len * sizeof(float), cudaMemcpyHostToDevice, stream);
        Launch(ctx.device + p * part, len / row_size, row_size, stream);
        cudaEventRecord(ctx.computed[p], stream);
        issued.store(p + 1, std::memory_order_release);
    }

    for (int i = 0; i < kStreams; ++i) {
        cudaEventRecord(ctx.idle[i], ctx.streams[i]);
    }
    for (int i = 0; i < kStreams; ++i) {
        cudaEventSynchronize(ctx.idle[i]);
    }
    cudaHostUnregister(in);

    for (size_t c = 0; c < chunks; ++c) {
        WaitFor(received, c);
        cudaEventSynchronize(ctx.received[c]);
        cudaHostUnregister(out + begin(c));
    }

    builder.join();
    receiver.join();
    return output;
}
