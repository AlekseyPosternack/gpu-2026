#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace {

void CheckCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(status));
    }
}

void CheckBlas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS Error status: " + std::to_string(static_cast<int>(status)));
    }
}

struct CublasContext {
    cublasHandle_t handle = nullptr;
    cudaStream_t stream = nullptr;
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    std::size_t capacity = 0;

    CublasContext() {
        CheckBlas(cublasCreate(&handle));
        CheckCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CheckBlas(cublasSetStream(handle, stream));
    }

    ~CublasContext() {
        if (stream) cudaStreamSynchronize(stream);
        if (d_a) {
            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_c);
        }
        if (handle) cublasDestroy(handle);
        if (stream) cudaStreamDestroy(stream);
    }
};

CublasContext& GetContext() {
    static thread_local CublasContext ctx;
    return ctx;
}

void EnsureBuffers(CublasContext& ctx, std::size_t need) {
    if (ctx.capacity >= need) {
        return;
    }
    if (ctx.d_a != nullptr) {
        CheckCuda(cudaFree(ctx.d_a));
        CheckCuda(cudaFree(ctx.d_b));
        CheckCuda(cudaFree(ctx.d_c));
    }
    const std::size_t bytes = need * sizeof(float);
    CheckCuda(cudaMalloc(&ctx.d_a, bytes));
    CheckCuda(cudaMalloc(&ctx.d_b, bytes));
    CheckCuda(cudaMalloc(&ctx.d_c, bytes));
    ctx.capacity = need;
}

} // namespace

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (n <= 0) {
        return {};
    }

    const std::size_t elements = static_cast<std::size_t>(n) * n;
    const std::size_t bytes = elements * sizeof(float);

    CublasContext& ctx = GetContext();
    EnsureBuffers(ctx, elements);

    CheckCuda(cudaMemcpyAsync(ctx.d_a, a.data(), bytes, cudaMemcpyHostToDevice, ctx.stream));
    CheckCuda(cudaMemcpyAsync(ctx.d_b, b.data(), bytes, cudaMemcpyHostToDevice, ctx.stream));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CheckBlas(cublasSgemm(ctx.handle,
                          CUBLAS_OP_N, CUBLAS_OP_N,
                          n, n, n,
                          &alpha,
                          ctx.d_b, n,
                          ctx.d_a, n,
                          &beta,
                          ctx.d_c, n));

    std::vector<float> output(elements);
    CheckCuda(cudaMemcpyAsync(output.data(), ctx.d_c, bytes, cudaMemcpyDeviceToHost, ctx.stream));
    CheckCuda(cudaStreamSynchronize(ctx.stream));

    return output;
}