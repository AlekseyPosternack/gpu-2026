#include "gemm_cublas.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace {

struct DeviceCache {
    cublasHandle_t handle = nullptr;
    cudaStream_t stream = nullptr;
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    std::size_t cap = 0;
};

DeviceCache& Cache() {
    static DeviceCache cache;
    return cache;
}

void Ensure(std::size_t n) {
    DeviceCache& cache = Cache();
    if (cache.handle == nullptr) {
        cublasCreate(&cache.handle);
        cudaStreamCreateWithFlags(&cache.stream, cudaStreamNonBlocking);
        cublasSetStream(cache.handle, cache.stream);
    }
    const std::size_t need = n * n;
    if (cache.cap >= need) {
        return;
    }
    if (cache.d_a != nullptr) {
        cudaFree(cache.d_a);
        cudaFree(cache.d_b);
        cudaFree(cache.d_c);
    }
    const std::size_t bytes = need * sizeof(float);
    cudaMalloc(&cache.d_a, bytes);
    cudaMalloc(&cache.d_b, bytes);
    cudaMalloc(&cache.d_c, bytes);
    cache.cap = need;
}

}

std::vector<float> GemmCUBLAS(const std::vector<float>& a, const std::vector<float>& b, int n) {
    if (n <= 0) {
        return {};
    }

    Ensure(static_cast<std::size_t>(n));
    DeviceCache& cache = Cache();
    const std::size_t bytes = static_cast<std::size_t>(n) * static_cast<std::size_t>(n) * sizeof(float);

    cudaMemcpyAsync(cache.d_a, a.data(), bytes, cudaMemcpyHostToDevice, cache.stream);
    cudaMemcpyAsync(cache.d_b, b.data(), bytes, cudaMemcpyHostToDevice, cache.stream);

    std::vector<float> output(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(cache.handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, cache.d_b, n, cache.d_a, n,
                &beta, cache.d_c, n);

    cudaMemcpyAsync(output.data(), cache.d_c, bytes, cudaMemcpyDeviceToHost, cache.stream);
    cudaStreamSynchronize(cache.stream);
    return output;
}
