#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <future>
#include <stdexcept>

namespace {

void CheckCUDA(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

void CheckCUBLAS(cublasStatus_t error) {
    if (error != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(cublasGetStatusString(error));
    }
}

struct Context {
    cublasHandle_t handle = nullptr;
    float* data = nullptr;
    std::size_t capacity = 0;
    Context() { CheckCUBLAS(cublasCreate(&handle)); }
    ~Context() {
        cublasDestroy(handle);
        cudaFree(data);
    }

    void Reserve(std::size_t count) {
        if (count > capacity) {
            float* next = nullptr;
            CheckCUDA(cudaMalloc(&next, 3 * count * sizeof(float)));
            cudaFree(data);
            data = next;
            capacity = count;
        }
    }
};
}

std::vector<float> GemmCUBLAS(const std::vector<float>& a, const std::vector<float>& b, int n) {
    if (n <= 0) {
        return {};
    }
    const std::size_t count = static_cast<std::size_t>(n) * n;
    if (a.size() != count || b.size() != count) {
        throw std::invalid_argument("Matrix size does not match n");
    }
    const std::size_t bytes = count * sizeof(float);
    thread_local Context context;
    context.Reserve(count);
    float* da = context.data;
    float* db = da + count;
    float* dc = db + count;
    auto output = std::async(std::launch::async, [count] { return std::vector<float>(count); });
    CheckCUDA(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
    CheckCUDA(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CheckCUBLAS(cublasSgemm(context.handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                           &alpha, db, n, da, n, &beta, dc, n));
    std::vector<float> result = output.get();
    CheckCUDA(cudaMemcpy(result.data(), dc, bytes, cudaMemcpyDeviceToHost));
    return result;
}
