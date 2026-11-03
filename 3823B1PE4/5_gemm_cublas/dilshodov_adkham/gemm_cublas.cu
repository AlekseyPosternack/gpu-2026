#include "gemm_cublas.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace {

// cuBLAS handle and device memory are created once and reused between calls
struct Context {
    cublasHandle_t handle = nullptr;
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
    size_t capacity = 0;

    Context() { cublasCreate(&handle); }

    void Reserve(size_t count) {
        if (count <= capacity) return;
        cudaFree(a);
        cudaFree(b);
        cudaFree(c);
        cudaMalloc(&a, count * sizeof(float));
        cudaMalloc(&b, count * sizeof(float));
        cudaMalloc(&c, count * sizeof(float));
        capacity = count;
    }
};

}  // namespace

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (n <= 0) return {};
    const size_t count = size_t(n) * n;
    const size_t bytes = count * sizeof(float);

    static Context ctx;
    ctx.Reserve(count);

    cudaMemcpy(ctx.a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx.b, b.data(), bytes, cudaMemcpyHostToDevice);

    // cuBLAS is column-major: a row-major matrix X is seen as X^T.
    // Row-major C = A * B  <=>  C^T = B^T * A^T, so pass B first and no transposition is needed.
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(ctx.handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                &alpha, ctx.b, n, ctx.a, n, &beta, ctx.c, n);

    // cuBLAS call is asynchronous, so the result is allocated on host while it runs
    std::vector<float> c(count);
    cudaMemcpy(c.data(), ctx.c, bytes, cudaMemcpyDeviceToHost);
    return c;
}
