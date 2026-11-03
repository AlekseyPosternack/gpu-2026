#include "gemm_cublas.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace {

// cuBLAS handle, stream and device buffers are created once and reused between
// calls, so no redundant device memory allocation happens during measurements.
struct CublasContext {
    cublasHandle_t handle = nullptr;
    cudaStream_t stream = nullptr;
    float* dev_a = nullptr;
    float* dev_b = nullptr;
    float* dev_c = nullptr;
    std::size_t capacity = 0;

    CublasContext() {
        cublasCreate(&handle);
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        cublasSetStream(handle, stream);
    }

    ~CublasContext() {
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
        if (handle != nullptr) cublasDestroy(handle);
        if (stream != nullptr) cudaStreamDestroy(stream);
    }

    void Reserve(std::size_t count) {
        if (count <= capacity) return;
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
        cudaMalloc(&dev_a, count * sizeof(float));
        cudaMalloc(&dev_b, count * sizeof(float));
        cudaMalloc(&dev_c, count * sizeof(float));
        capacity = count;
    }
};

CublasContext& GetContext() {
    static CublasContext context;
    return context;
}

}  // namespace

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (n <= 0) return {};
    const std::size_t count = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    if (a.size() < count || b.size() < count) return {};
    const std::size_t bytes = count * sizeof(float);

    CublasContext& ctx = GetContext();
    ctx.Reserve(count);

    // Both input matrices are stored by rows on the host.
    cudaMemcpyAsync(ctx.dev_a, a.data(), bytes, cudaMemcpyHostToDevice, ctx.stream);
    cudaMemcpyAsync(ctx.dev_b, b.data(), bytes, cudaMemcpyHostToDevice, ctx.stream);

    // cuBLAS works with column-major matrices: a row-major matrix X is seen as
    // X^T. Therefore row-major C = A * B is the same as column-major
    // C^T = B^T * A^T, so passing B and then A avoids any explicit transpose.
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(ctx.handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                &alpha, ctx.dev_b, n, ctx.dev_a, n, &beta, ctx.dev_c, n);

    // The cuBLAS call is asynchronous, so the result buffer is allocated on the
    // host while the GPU is still doing the multiplication.
    std::vector<float> result(count);
    cudaMemcpyAsync(result.data(), ctx.dev_c, bytes, cudaMemcpyDeviceToHost, ctx.stream);
    cudaStreamSynchronize(ctx.stream);
    return result;
}
