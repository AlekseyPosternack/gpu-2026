#include "gelu_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>

namespace {

constexpr float kK1 = -1.5957691216057308f;
constexpr float kK2 = -0.0713548162726009f;
constexpr size_t kChunk = size_t(1) << 22;
constexpr size_t kSmall = size_t(1) << 20;
constexpr size_t kPage = 4096;
constexpr int kStreams = 4;
constexpr int kBlock = 256;

__device__ __forceinline__ float Gelu(float x) {
    return x / (1.0f + __expf(x * (kK1 + kK2 * x * x)));
}

__global__ void GeluKernel(float* data, size_t n) {
    const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    const size_t n4 = n / 4;
    if (i < n4) {
        float4 v = reinterpret_cast<float4*>(data)[i];
        v.x = Gelu(v.x);
        v.y = Gelu(v.y);
        v.z = Gelu(v.z);
        v.w = Gelu(v.w);
        reinterpret_cast<float4*>(data)[i] = v;
    } else if (i < n4 + n % 4) {
        const size_t j = 3 * n4 + i;
        data[j] = Gelu(data[j]);
    }
}

void Launch(float* data, size_t n, cudaStream_t stream) {
    const unsigned blocks = static_cast<unsigned>((n / 4 + n % 4 + kBlock - 1) / kBlock);
    GeluKernel<<<blocks, kBlock, 0, stream>>>(data, n);
}

struct Context {
    float* device = nullptr;
    size_t capacity = 0;
    cudaStream_t streams[kStreams];
    std::vector<cudaEvent_t> events;

    Context() {
        for (cudaStream_t& s : streams) {
            cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        }
    }

    ~Context() {
        cudaFree(device);
        for (cudaStream_t s : streams) {
            cudaStreamDestroy(s);
        }
        for (cudaEvent_t e : events) {
            cudaEventDestroy(e);
        }
    }

    void Reserve(size_t n) {
        if (n > capacity) {
            cudaFree(device);
            cudaMalloc(&device, n * sizeof(float));
            capacity = n;
        }
        while (events.size() < n / kChunk + 2 + kStreams) {
            cudaEvent_t e;
            cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
            events.push_back(e);
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

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const size_t n = input.size();
    Context& ctx = GetContext();
    ctx.Reserve(n);

    if (n < kSmall) {
        std::vector<float> output(n);
        if (n > 0) {
            cudaMemcpy(ctx.device, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);
            Launch(ctx.device, n, 0);
            cudaMemcpy(output.data(), ctx.device, n * sizeof(float), cudaMemcpyDeviceToHost);
        }
        return output;
    }

    std::vector<float> output;
    output.reserve(n);
    float* out = output.data();
    const size_t shift = reinterpret_cast<uintptr_t>(out) % kPage / sizeof(float);
    const size_t chunks = (n + shift + kChunk - 1) / kChunk;
    auto begin = [&](size_t c) { return c == 0 ? 0 : c * kChunk - shift; };
    auto end = [&](size_t c) { return std::min(n, (c + 1) * kChunk - shift); };

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
            cudaStream_t stream = ctx.streams[c % kStreams];
            WaitFor(built, c);
            cudaHostRegister(out + begin(c), bytes, cudaHostRegisterDefault);
            WaitFor(issued, c);
            cudaMemcpyAsync(out + begin(c), ctx.device + begin(c), bytes, cudaMemcpyDeviceToHost, stream);
            cudaEventRecord(ctx.events[c], stream);
            received.store(c + 1, std::memory_order_release);
        }
    });

    float* in = const_cast<float*>(input.data());
    cudaHostRegister(in, n * sizeof(float), cudaHostRegisterDefault);
    for (size_t c = 0; c < chunks; ++c) {
        const size_t len = end(c) - begin(c);
        cudaStream_t stream = ctx.streams[c % kStreams];
        cudaMemcpyAsync(ctx.device + begin(c), in + begin(c), len * sizeof(float), cudaMemcpyHostToDevice, stream);
        Launch(ctx.device + begin(c), len, stream);
        issued.store(c + 1, std::memory_order_release);
    }

    for (int i = 0; i < kStreams; ++i) {
        cudaEventRecord(ctx.events[chunks + i], ctx.streams[i]);
    }
    for (int i = 0; i < kStreams; ++i) {
        cudaEventSynchronize(ctx.events[chunks + i]);
    }
    cudaHostUnregister(in);

    for (size_t c = 0; c < chunks; ++c) {
        WaitFor(received, c);
        cudaEventSynchronize(ctx.events[c]);
        cudaHostUnregister(out + begin(c));
    }

    builder.join();
    receiver.join();
    return output;
}
