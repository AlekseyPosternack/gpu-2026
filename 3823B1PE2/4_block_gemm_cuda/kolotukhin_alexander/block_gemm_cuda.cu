#include "block_gemm_cuda.h"

#include <cstdio>
#include <cstdlib>
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
constexpr int BK = 16;

constexpr int TM = 8;
constexpr int TN = 8;

__global__ void block_gemm_kernel_8x8(const float* __restrict__ a,
                                       const float* __restrict__ b,
                                       float* __restrict__ c,
                                       int n) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int thread_row = ty * TM;
    const int thread_col = tx * TN;
    
    const int row_a = by * BM;
    const int col_b = bx * BN;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float acc[TM][TN] = {};

    for (int k_tile = 0; k_tile < n; k_tile += BK) {
        
        #pragma unroll
        for (int i = 0; i < (BM * BK) / 256; i++) {
            int idx = ty * 16 + tx + i * 256;
            int r = idx / BK;
            int c = idx % BK;
            As[r][c] = __ldg(&a[(row_a + r) * n + k_tile + c]);
        }

        #pragma unroll
        for (int i = 0; i < (BK * BN) / 256; i++) {
            int idx = ty * 16 + tx + i * 256;
            int r = idx / BN;
            int c = idx % BN;
            Bs[r][c] = __ldg(&b[(k_tile + r) * n + col_b + c]);
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float a_frag[TM];
            float b_frag[TN];

            #pragma unroll
            for (int i = 0; i < TM; i++) {
                a_frag[i] = As[thread_row + i][k];
            }
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                b_frag[j] = Bs[k][thread_col + j];
            }

            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    acc[i][j] = fmaf(a_frag[i], b_frag[j], acc[i][j]);
                }
            }
        }
        
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            c[(row_a + thread_row + i) * n + col_b + thread_col + j] = acc[i][j];
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

Cache& getCache() {
    static Cache c;
    return c;
}

}  // namespace

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                  const std::vector<float>& b,
                                  int n) {
    if (n <= 0) return {};

    const std::size_t total_elements = static_cast<std::size_t>(n) * n;
    const std::size_t bytes = total_elements * sizeof(float);

    std::vector<float> c;
    std::thread alloc_thread([&c, total_elements]() {
        c.resize(total_elements);
    });

    Cache& cache_ = getCache();
    cache_.init(static_cast<std::size_t>(n));

    CHECK_ERROR(cudaMemcpy(cache_.d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(cache_.d_b, b.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(BN / TN, BM / TM);
    dim3 grid((n + BN - 1) / BN, (n + BM - 1) / BM);

    block_gemm_kernel_8x8<<<grid, block>>>(cache_.d_a, cache_.d_b, cache_.d_c, n);
    CHECK_ERROR(cudaGetLastError());

    if (alloc_thread.joinable()) {
        alloc_thread.join();
    }

    CHECK_ERROR(cudaDeviceSynchronize());
    CHECK_ERROR(cudaMemcpy(c.data(), cache_.d_c, bytes, cudaMemcpyDeviceToHost));

    return c;
}