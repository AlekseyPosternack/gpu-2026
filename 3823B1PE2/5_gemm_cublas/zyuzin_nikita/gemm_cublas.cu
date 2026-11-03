#include "gemm_cublas.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <thread>

namespace {

struct Context {
    cublasHandle_t handle = nullptr;
    cudaStream_t stream = nullptr;
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
    size_t capacity = 0;

    Context() {
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        cublasCreate(&handle);
        cublasSetStream(handle, stream);
    }

    void Reserve(size_t count) {
        if (count <= capacity) {
            return;
        }
        cudaFree(a);
        cudaFree(b);
        cudaFree(c);
        cudaMalloc(&a, count * sizeof(float));
        cudaMalloc(&b, count * sizeof(float));
        cudaMalloc(&c, count * sizeof(float));
        capacity = count;
    }
};

Context& GetContext() {
    static Context ctx;
    return ctx;
}

}  // namespace

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (n <= 0) {
        return {};
    }
    const size_t count = static_cast<size_t>(n) * n;
    const size_t bytes = count * sizeof(float);

    std::vector<float> c;
    std::thread allocator([&c, count] { c = std::vector<float>(count); });

    Context& ctx = GetContext();
    ctx.Reserve(count);

    cudaMemcpyAsync(ctx.a, a.data(), bytes, cudaMemcpyHostToDevice, ctx.stream);
    cudaMemcpyAsync(ctx.b, b.data(), bytes, cudaMemcpyHostToDevice, ctx.stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(ctx.handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                &alpha, ctx.b, n, ctx.a, n, &beta, ctx.c, n);

    allocator.join();
    cudaMemcpyAsync(c.data(), ctx.c, bytes, cudaMemcpyDeviceToHost, ctx.stream);
    cudaStreamSynchronize(ctx.stream);

    return c;
}
