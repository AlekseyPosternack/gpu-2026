#include "gemm_cublas.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace {

constexpr float kAlpha = 1.0f;
constexpr float kBeta  = 0.0f;

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

void check_cublas(cublasStatus_t st, const char* what) {
    if (st != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(what) + ": cublas error " +
                                 std::to_string(static_cast<int>(st)));
    }
}

struct DeviceBuffer {
    float* ptr = nullptr;
    std::size_t cap = 0;

    void ensure(std::size_t n) {
        if (n > cap) {
            if (ptr) check_cuda(cudaFree(ptr), "cudaFree");
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(float)),
                       "cudaMalloc");
            cap = n;
        }
    }

    ~DeviceBuffer() {
        if (ptr) cudaFree(ptr);
    }
};

cublasHandle_t& get_handle() {
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        check_cublas(cublasCreate(&handle), "cublasCreate");
        check_cublas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST),
                     "cublasSetPointerMode");
    }
    return handle;
}

} // namespace

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    const std::size_t total = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    std::vector<float> c(total);
    if (total == 0) return c;

    static DeviceBuffer d_a;
    static DeviceBuffer d_b;
    static DeviceBuffer d_c;

    d_a.ensure(total);
    d_b.ensure(total);
    d_c.ensure(total);

    const std::size_t bytes = total * sizeof(float);

    check_cuda(cudaMemcpyAsync(d_a.ptr, a.data(), bytes, cudaMemcpyHostToDevice, 0),
               "cudaMemcpy a H2D");
    check_cuda(cudaMemcpyAsync(d_b.ptr, b.data(), bytes, cudaMemcpyHostToDevice, 0),
               "cudaMemcpy b H2D");

    cublasHandle_t handle = get_handle();

    check_cublas(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             n, n, n,
                             &kAlpha,
                             d_b.ptr, n,
                             d_a.ptr, n,
                             &kBeta,
                             d_c.ptr, n),
                 "cublasSgemm");

    check_cuda(cudaMemcpyAsync(c.data(), d_c.ptr, bytes, cudaMemcpyDeviceToHost, 0),
               "cudaMemcpy c D2H");

    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    return c;
}