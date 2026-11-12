#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstddef>

namespace {

struct CublasWorker {
    cublasHandle_t handle = nullptr;
    cudaStream_t stream = nullptr;
    float* device_memory = nullptr;
    size_t allocated_elements = 0;

    CublasWorker() {
        cublasCreate(&handle);
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        cublasSetStream(handle, stream);
    }

    ~CublasWorker() {
        if (device_memory != nullptr) {
            cudaFree(device_memory);
        }
        if (handle != nullptr) {
            cublasDestroy(handle);
        }
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
    }

    float* RequestMemory(size_t elements_per_matrix) {
        const size_t total_required = 3 * elements_per_matrix;
        if (total_required > allocated_elements) {
            if (device_memory != nullptr) {
                cudaFree(device_memory);
            }
            cudaMalloc(&device_memory, total_required * sizeof(float));
            allocated_elements = total_required;
        }
        return device_memory;
    }

    CublasWorker(const CublasWorker&) = delete;
    CublasWorker& operator=(const CublasWorker&) = delete;
};

} // namespace

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (n <= 0) {
        return {};
    }

    static thread_local CublasWorker worker;

    const size_t matrix_elements = static_cast<size_t>(n) * n;
    const size_t matrix_bytes = matrix_elements * sizeof(float);

    float* base_ptr = worker.RequestMemory(matrix_elements);
    float* d_a = base_ptr;
    float* d_b = base_ptr + matrix_elements;
    float* d_c = base_ptr + 2 * matrix_elements;

    cudaMemcpyAsync(d_a, a.data(), matrix_bytes, cudaMemcpyHostToDevice, worker.stream);
    cudaMemcpyAsync(d_b, b.data(), matrix_bytes, cudaMemcpyHostToDevice, worker.stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(
        worker.handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n,
        &alpha,
        d_b, n,
        d_a, n,
        &beta,
        d_c, n
    );

    std::vector<float> result(matrix_elements);

    cudaMemcpyAsync(result.data(), d_c, matrix_bytes, cudaMemcpyDeviceToHost, worker.stream);

    cudaStreamSynchronize(worker.stream);

    return result;
}