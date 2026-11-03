#include "naive_gemm_cuda.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <cuda_runtime.h>

#define CHECK_ERROR(X)                                          \
    do {                                                        \
        cudaError_t error = (X);                                \
        if (error != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error at %s:%d — %s: %s\n",   \
                    __FILE__, __LINE__,                         \
                    cudaGetErrorName(error),                    \
                    cudaGetErrorString(error));                 \
            std::exit(EXIT_FAILURE);                            \
        }                                                       \
    } while (0)

namespace {

constexpr int BM = 128;
constexpr int BN = 128;

constexpr int TM = 8;
constexpr int TN = 8;

__global__ void naive_gemm_kernel(const float* __restrict__ a,
                                   const float* __restrict__ b,
                                   float* __restrict__ c,
                                   int n) {
    const int thread_row = threadIdx.y * TM;
    const int thread_col = threadIdx.x * TN;
    const int block_row  = blockIdx.y * BM;
    const int block_col  = blockIdx.x * BN;

    float acc[TM][TN] = {};

    const float* a_row = a + (block_row + thread_row) * n;
    const float* b_col = b + block_col + thread_col;

    for (int k = 0; k < n; ++k) {
        float a_reg[TM];
        float b_reg[TN];

        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            a_reg[i] = a_row[i * n + k];
        }

        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            b_reg[j] = b_col[k * n + j];
        }

        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                acc[i][j] = fmaf(a_reg[i], b_reg[j], acc[i][j]);
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            c[(block_row + thread_row + i) * n + block_col + thread_col + j] = acc[i][j];
        }
    }
}

struct Cache {
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    std::size_t capacity = 0;

    ~Cache() {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
    }

    void init(std::size_t n) {
        if (n <= capacity) return;
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
        const std::size_t bytes = n * n * sizeof(float);
        CHECK_ERROR(cudaMalloc(&d_a, bytes));
        CHECK_ERROR(cudaMalloc(&d_b, bytes));
        CHECK_ERROR(cudaMalloc(&d_c, bytes));
        capacity = n;
    }
};

Cache& getCache() { static Cache c; return c; }

}  // namespace

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) {
        return {};
    }

    const std::size_t total = static_cast<std::size_t>(n) * n;
    const std::size_t bytes = total * sizeof(float);

    std::vector<float> c;
    std::thread alloc([&c, total] { c.resize(total); });

    Cache& cache_ = getCache();
    cache_.init(n);

    CHECK_ERROR(cudaMemcpy(cache_.d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(cache_.d_b, b.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(BN / TN, BM / TM);
    dim3 grid((n + BN - 1) / BN, (n + BM - 1) / BM);
    naive_gemm_kernel<<<grid, block>>>(cache_.d_a, cache_.d_b, cache_.d_c, n);
    CHECK_ERROR(cudaGetLastError());

    alloc.join();   // ждём аллокацию (она уже параллельно прошла)

    CHECK_ERROR(cudaDeviceSynchronize());
    CHECK_ERROR(cudaMemcpy(c.data(), cache_.d_c, bytes, cudaMemcpyDeviceToHost));

    return c;
}