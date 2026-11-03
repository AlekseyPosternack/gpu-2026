#include "gemm_cublas.h"

#include <cstdio>
#include <cstdlib>
#include <thread>

#include <cublas_v2.h>
#include <cuda_runtime.h>


#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                        \
            std::fprintf(stderr, "CUDA error \"%s\" at %s:%d\n",          \
                         cudaGetErrorString(err_), __FILE__, __LINE__);   \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

#define CUBLAS_CHECK(call)                                                \
    do {                                                                  \
        cublasStatus_t st_ = (call);                                      \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                               \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n",            \
                         static_cast<int>(st_), __FILE__, __LINE__);      \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

namespace {

    
    class GemmResources {
    public:
        static GemmResources& Instance() {
            
            static GemmResources resources;
            return resources;
        }

        cublasHandle_t Handle() const { return handle_; }

        void Reserve(size_t count) {
            if (count <= capacity_) {
                return;
            }
            Free();
            CUDA_CHECK(cudaMalloc(&a_, count * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&b_, count * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&c_, count * sizeof(float)));
            capacity_ = count;
        }

        float* A() const { return a_; }
        float* B() const { return b_; }
        float* C() const { return c_; }

        GemmResources(const GemmResources&) = delete;
        GemmResources& operator=(const GemmResources&) = delete;

    private:
        GemmResources() { CUBLAS_CHECK(cublasCreate(&handle_)); }

        ~GemmResources() {
            Free();
            cublasDestroy(handle_);
        }

        void Free() {
           
            if (a_ != nullptr) cudaFree(a_);
            if (b_ != nullptr) cudaFree(b_);
            if (c_ != nullptr) cudaFree(c_);
            a_ = b_ = c_ = nullptr;
            capacity_ = 0;
        }

        cublasHandle_t handle_ = nullptr;
        float* a_ = nullptr;
        float* b_ = nullptr;
        float* c_ = nullptr;
        size_t capacity_ = 0;
    };

}  // namespace

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    if (n <= 0) {
        return {};
    }
    const size_t count = static_cast<size_t>(n) * n;
    const size_t bytes = count * sizeof(float);

    GemmResources& res = GemmResources::Instance();
    res.Reserve(count);

    
    std::vector<float> c;
    std::thread alloc_thread([&c, count] { c.resize(count); });

    CUDA_CHECK(cudaMemcpy(res.A(), a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(res.B(), b.data(), bytes, cudaMemcpyHostToDevice));

  
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(res.Handle(),
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n,
        &alpha,
        res.B(), n,
        res.A(), n,
        &beta,
        res.C(), n));

    alloc_thread.join();

    
    CUDA_CHECK(cudaMemcpy(c.data(), res.C(), bytes, cudaMemcpyDeviceToHost));

    return c;
}