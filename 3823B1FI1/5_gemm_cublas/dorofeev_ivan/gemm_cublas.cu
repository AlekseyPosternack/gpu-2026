#include "gemm_cublas.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <thread>

namespace {

constexpr int kStrips = 8;
constexpr int kMinRows = 128;

struct Context {
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
    size_t capacity = 0;
    cudaStream_t copy_stream;
    cudaStream_t streams[kStrips];
    cublasHandle_t handles[kStrips];
    cudaEvent_t a_ready[kStrips];
    cudaEvent_t b_ready[2];

    Context() {
        cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking);
        for (int i = 0; i < kStrips; ++i) {
            cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
            cudaEventCreateWithFlags(&a_ready[i], cudaEventDisableTiming);
            cublasCreate(&handles[i]);
            cublasSetStream(handles[i], streams[i]);
        }
        for (cudaEvent_t& e : b_ready) {
            cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
        }
    }

    ~Context() {
        cudaFree(a);
        cudaFree(b);
        cudaFree(c);
        for (int i = 0; i < kStrips; ++i) {
            cublasDestroy(handles[i]);
            cudaStreamDestroy(streams[i]);
            cudaEventDestroy(a_ready[i]);
        }
        cudaStreamDestroy(copy_stream);
        for (cudaEvent_t e : b_ready) {
            cudaEventDestroy(e);
        }
    }

    void Reserve(size_t count) {
        if (count > capacity) {
            cudaFree(a);
            cudaFree(b);
            cudaFree(c);
            cudaMalloc(&a, count * sizeof(float));
            cudaMalloc(&b, count * sizeof(float));
            cudaMalloc(&c, count * sizeof(float));
            capacity = count;
        }
    }
};

Context& GetContext() {
    static Context context;
    return context;
}

void Multiply(cublasHandle_t handle, const float* a, const float* b, float* c, int n, int rows, int k_begin,
              int k_end, float beta) {
    const float alpha = 1.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, rows, k_end - k_begin, &alpha, b + size_t(k_begin) * n, n,
                a + k_begin, n, &beta, c, n);
}

}

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    const size_t count = size_t(n) * n;
    const size_t bytes = count * sizeof(float);
    Context& ctx = GetContext();
    ctx.Reserve(count);

    const int rows = n / kStrips;
    if (rows < kMinRows) {
        std::vector<float> c(count);
        if (n > 0) {
            cudaStream_t stream = ctx.streams[0];
            cudaMemcpyAsync(ctx.a, a.data(), bytes, cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(ctx.b, b.data(), bytes, cudaMemcpyHostToDevice, stream);
            Multiply(ctx.handles[0], ctx.a, ctx.b, ctx.c, n, n, 0, n, 0.0f);
            cudaMemcpyAsync(c.data(), ctx.c, bytes, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        }
        return c;
    }

    std::vector<float> c;
    std::thread builder([&] { c.resize(count); });

    const size_t strip = size_t(rows) * n;
    const int half = n / 2;
    const size_t half_size = size_t(half) * n;

    auto copy_b = [&](int p) {
        cudaMemcpyAsync(ctx.b + p * half_size, b.data() + p * half_size, half_size * sizeof(float),
                        cudaMemcpyHostToDevice, ctx.copy_stream);
        cudaEventRecord(ctx.b_ready[p], ctx.copy_stream);
    };
    auto launch = [&](int p, int r) {
        cudaStreamWaitEvent(ctx.streams[r], ctx.b_ready[p], 0);
        Multiply(ctx.handles[r], ctx.a + r * strip, ctx.b, ctx.c + r * strip, n, rows, p * half, (p + 1) * half,
                 p == 0 ? 0.0f : 1.0f);
    };

    const int middle = (kStrips - 1) / 2;
    copy_b(0);
    for (int r = 0; r < kStrips; ++r) {
        cudaMemcpyAsync(ctx.a + r * strip, a.data() + r * strip, strip * sizeof(float), cudaMemcpyHostToDevice,
                        ctx.copy_stream);
        cudaEventRecord(ctx.a_ready[r], ctx.copy_stream);
        cudaStreamWaitEvent(ctx.streams[r], ctx.a_ready[r], 0);
        launch(0, r);
        if (r == middle) {
            copy_b(1);
            for (int q = 0; q <= r; ++q) {
                launch(1, q);
            }
        } else if (r > middle) {
            launch(1, r);
        }
    }

    builder.join();
    for (int r = 0; r < kStrips; ++r) {
        cudaMemcpyAsync(c.data() + r * strip, ctx.c + r * strip, strip * sizeof(float), cudaMemcpyDeviceToHost,
                        ctx.streams[r]);
    }
    for (int r = 0; r < kStrips; ++r) {
        cudaStreamSynchronize(ctx.streams[r]);
    }
    return c;
}
