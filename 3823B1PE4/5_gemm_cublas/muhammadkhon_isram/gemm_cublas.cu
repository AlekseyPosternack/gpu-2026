#include "gemm_cublas.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstddef>

namespace {

struct Context {
    cublasHandle_t handle = nullptr;
    cudaStream_t stream = nullptr;
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    size_t capacity = 0;

    Context() {
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        cublasCreate(&handle);
        cublasSetStream(handle, stream);
    }
    ~Context() {
        Free();
        cublasDestroy(handle);
        cudaStreamDestroy(stream);
    }
    void Free() {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
        d_a = d_b = d_c = nullptr;
        capacity = 0;
    }
    void Reserve(size_t elems) {
        if (elems <= capacity) return;
        Free();
        cudaMalloc(&d_a, elems * sizeof(float));
        cudaMalloc(&d_b, elems * sizeof(float));
        cudaMalloc(&d_c, elems * sizeof(float));
        capacity = elems;
    }
};

}  // namespace

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (n <= 0) return {};

    const size_t elems = static_cast<size_t>(n) * n;

    static Context ctx;
    ctx.Reserve(elems);

    cudaMemcpyAsync(ctx.d_a, a.data(), elems * sizeof(float),
                    cudaMemcpyHostToDevice, ctx.stream);
    cudaMemcpyAsync(ctx.d_b, b.data(), elems * sizeof(float),
                    cudaMemcpyHostToDevice, ctx.stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(ctx.handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha,
                ctx.d_b, n,
                ctx.d_a, n,
                &beta,
                ctx.d_c, n);

    std::vector<float> c(elems);

    cudaMemcpyAsync(c.data(), ctx.d_c, elems * sizeof(float),
                    cudaMemcpyDeviceToHost, ctx.stream);
    cudaStreamSynchronize(ctx.stream);

    return c;
}