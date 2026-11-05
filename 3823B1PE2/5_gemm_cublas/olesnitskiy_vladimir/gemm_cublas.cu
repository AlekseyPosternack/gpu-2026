#include "gemm_cublas.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace {

struct CublasContext {
    cublasHandle_t handle = nullptr;
    cudaStream_t stream = nullptr;
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    size_t capacity = 0;

    void Init(size_t count) {
        if (!handle) {
            cudaStreamCreate(&stream);
            cublasCreate(&handle);
            cublasSetStream(handle, stream);
        }
        if (count <= capacity) {
            return;
        }
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        cudaMalloc(&d_a, count * sizeof(float));
        cudaMalloc(&d_b, count * sizeof(float));
        cudaMalloc(&d_c, count * sizeof(float));
        capacity = count;
    }

    ~CublasContext() {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        if (handle) cublasDestroy(handle);
        if (stream) cudaStreamDestroy(stream);
    }
};

CublasContext ctx;

}  // namespace

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (n <= 0) {
        return {};
    }
    const size_t count = static_cast<size_t>(n) * n;
    const size_t bytes = count * sizeof(float);
    ctx.Init(count);

    cudaMemcpyAsync(ctx.d_a, a.data(), bytes, cudaMemcpyHostToDevice, ctx.stream);
    cudaMemcpyAsync(ctx.d_b, b.data(), bytes, cudaMemcpyHostToDevice, ctx.stream);

    // Row-major C = A * B is column-major C^T = B^T * A^T,
    // so passing B and A without transposition gives C in row-major order
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(ctx.handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                &alpha, ctx.d_b, n, ctx.d_a, n, &beta, ctx.d_c, n);

    std::vector<float> c(count);
    cudaMemcpyAsync(c.data(), ctx.d_c, bytes, cudaMemcpyDeviceToHost, ctx.stream);
    cudaStreamSynchronize(ctx.stream);
    return c;
}
