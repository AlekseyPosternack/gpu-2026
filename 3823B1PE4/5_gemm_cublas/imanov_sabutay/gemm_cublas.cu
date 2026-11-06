#include "gemm_cublas.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {

void CheckCUDA(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void CheckCUBLAS(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": cuBLAS status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

struct Context {
    cublasHandle_t handle = nullptr;
    float* data = nullptr;
    std::size_t capacity = 0;
    int device = 0;

    Context() {
        CheckCUDA(cudaGetDevice(&device), "cudaGetDevice");
        CheckCUBLAS(cublasCreate(&handle), "cublasCreate");
    }
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    ~Context() {
        int previous = 0;
        if (cudaGetDevice(&previous) != cudaSuccess) return;
        if (previous != device && cudaSetDevice(device) != cudaSuccess) return;
        if (handle != nullptr) cublasDestroy(handle);
        if (data != nullptr) cudaFree(data);
        if (previous != device) cudaSetDevice(previous);
    }

    void Reserve(std::size_t elements) {
        if (elements <= capacity) return;
        float* replacement = nullptr;
        CheckCUDA(cudaMalloc(reinterpret_cast<void**>(&replacement),
                             3 * elements * sizeof(float)), "cudaMalloc");
        if (data != nullptr) {
            const cudaError_t status = cudaFree(data);
            if (status != cudaSuccess) {
                cudaFree(replacement);
                CheckCUDA(status, "cudaFree");
            }
        }
        data = replacement;
        capacity = elements;
    }
};

} // namespace

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                            const std::vector<float>& b,
                            int n) {
    if (n < 0) throw std::invalid_argument("matrix size must be nonnegative");
    const std::size_t side = static_cast<std::size_t>(n);
    const std::size_t max_elements = std::numeric_limits<std::size_t>::max() / (3 * sizeof(float));
    if (side != 0 && side > max_elements / side) {
        throw std::length_error("matrix size is too large");
    }
    const std::size_t elements = side * side;
    if (a.size() != elements || b.size() != elements) {
        throw std::invalid_argument("input matrices must contain n*n elements");
    }
    if (elements == 0) return {};

    int device = 0;
    CheckCUDA(cudaGetDevice(&device), "cudaGetDevice");
    // Reuse a handle and allocation per host thread and CUDA device.
    thread_local std::unordered_map<int, Context> contexts;
    Context& context = contexts[device];
    context.Reserve(elements);
    float* device_a = context.data;
    float* device_b = device_a + elements;
    float* device_c = device_b + elements;
    const std::size_t bytes = elements * sizeof(float);
    CheckCUDA(cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice), "copy A to GPU");
    CheckCUDA(cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice), "copy B to GPU");

    const float alpha = 1.0f;
    const float beta = 0.0f;
    // Row-major C = A*B is column-major C^T = B^T*A^T.
    // Swap operands to avoid transposing or allocating temporary matrices.
    CheckCUBLAS(cublasSgemm(context.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                           n, n, n, &alpha, device_b, n, device_a, n,
                           &beta, device_c, n), "cublasSgemm");

    // Host allocation can overlap asynchronous cuBLAS execution.
    std::vector<float> output(elements);
    // Blocking copy waits for cuBLAS and reports CUDA execution errors.
    CheckCUDA(cudaMemcpy(output.data(), device_c, bytes, cudaMemcpyDeviceToHost), "copy C from GPU");
    return output;
}
