#include "block_gemm_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstring>

namespace {

constexpr int kTile = 32;

__global__ void BlockGemmKernel(const float* __restrict__ a, const float* __restrict__ b,
                                float* __restrict__ c, int n) {
    __shared__ float as[kTile][kTile + 1];
    __shared__ float bs[kTile][kTile + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * kTile + ty;
    const int col = blockIdx.x * kTile + tx;

    float sum = 0.0f;
    const int tiles = n / kTile;

    for (int t = 0; t < tiles; ++t) {
        as[ty][tx] = __ldg(a + row * n + t * kTile + tx);
        bs[ty][tx] = __ldg(b + (t * kTile + ty) * n + col);
        __syncthreads();

#pragma unroll
        for (int k = 0; k < kTile; ++k) {
            sum = fmaf(as[ty][k], bs[k][tx], sum);
        }
        __syncthreads();
    }

    c[row * n + col] = sum;
}

constexpr int kBM = 64;
constexpr int kBN = 64;
constexpr int kBK = 16;
constexpr int kTM = 4;
constexpr int kTN = 4;

__global__ void __launch_bounds__(256)
BlockGemmFastKernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c,
                    int n) {
    __shared__ __align__(16) float as[kBM][kBK];
    __shared__ __align__(16) float bs[kBK][kBN];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int a_row = tid / (kBK / 4);
    const int a_k = (tid % (kBK / 4)) * 4;
    const int b_k = tid / (kBN / 4);
    const int b_col = (tid % (kBN / 4)) * 4;

    float acc[kTM][kTN];
#pragma unroll
    for (int i = 0; i < kTM; ++i) {
#pragma unroll
        for (int j = 0; j < kTN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int t = 0; t < n; t += kBK) {
        const float4 va = __ldg(reinterpret_cast<const float4*>(
            a + (blockIdx.y * kBM + a_row) * n + t + a_k));
        as[a_row][a_k + 0] = va.x;
        as[a_row][a_k + 1] = va.y;
        as[a_row][a_k + 2] = va.z;
        as[a_row][a_k + 3] = va.w;
        const float4 vb = __ldg(reinterpret_cast<const float4*>(
            b + (t + b_k) * n + blockIdx.x * kBN + b_col));
        bs[b_k][b_col + 0] = vb.x;
        bs[b_k][b_col + 1] = vb.y;
        bs[b_k][b_col + 2] = vb.z;
        bs[b_k][b_col + 3] = vb.w;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < kBK; ++k) {
            float av[kTM];
            float bv[kTN];
#pragma unroll
            for (int i = 0; i < kTM; ++i) {
                av[i] = as[ty * kTM + i][k];
            }
#pragma unroll
            for (int j = 0; j < kTN; ++j) {
                bv[j] = bs[k][tx * kTN + j];
            }
#pragma unroll
            for (int i = 0; i < kTM; ++i) {
#pragma unroll
                for (int j = 0; j < kTN; ++j) {
                    acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

    const int row0 = blockIdx.y * kBM + ty * kTM;
    const int col0 = blockIdx.x * kBN + tx * kTN;
#pragma unroll
    for (int i = 0; i < kTM; ++i) {
        *reinterpret_cast<float4*>(c + (row0 + i) * n + col0) =
            make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
    }
}

__global__ void BlockGemmSmallKernel(const float* __restrict__ a, const float* __restrict__ b,
                                     float* __restrict__ c, int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) {
        return;
    }
    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
        sum = fmaf(a[row * n + k], b[k * n + col], sum);
    }
    c[row * n + col] = sum;
}

struct DeviceCache {
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    float* h_pin_c = nullptr;
    const float* registered_a = nullptr;
    const float* registered_b = nullptr;
    cudaStream_t stream = nullptr;
    std::size_t cap = 0;
    std::size_t pin_cap = 0;
};

DeviceCache& Cache() {
    static DeviceCache cache;
    return cache;
}

void RegisterPtr(const float*& slot, const float* ptr, std::size_t bytes) {
    if (slot == ptr) {
        return;
    }
    if (slot != nullptr) {
        cudaHostUnregister(const_cast<float*>(slot));
        slot = nullptr;
    }
    if (cudaHostRegister(const_cast<float*>(ptr), bytes, cudaHostRegisterDefault) == cudaSuccess) {
        slot = ptr;
    }
}

void Ensure(std::size_t n) {
    DeviceCache& cache = Cache();
    if (cache.stream == nullptr) {
        cudaStreamCreateWithFlags(&cache.stream, cudaStreamNonBlocking);
    }
    const std::size_t need = n * n;
    if (cache.cap < need) {
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
    if (cache.pin_cap < need) {
        if (cache.h_pin_c != nullptr) {
            cudaFreeHost(cache.h_pin_c);
        }
        cudaMallocHost(&cache.h_pin_c, need * sizeof(float));
        cache.pin_cap = need;
    }
}

}  // namespace

std::vector<float> BlockGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    if (n <= 0) {
        return {};
    }

    Ensure(static_cast<std::size_t>(n));
    DeviceCache& cache = Cache();
    const std::size_t bytes = static_cast<std::size_t>(n) * static_cast<std::size_t>(n) * sizeof(float);
    RegisterPtr(cache.registered_a, a.data(), bytes);
    RegisterPtr(cache.registered_b, b.data(), bytes);

    cudaMemcpyAsync(cache.d_a, a.data(), bytes, cudaMemcpyHostToDevice, cache.stream);
    cudaMemcpyAsync(cache.d_b, b.data(), bytes, cudaMemcpyHostToDevice, cache.stream);

    if (n >= kBM && (n % kBM) == 0 && (n % kBN) == 0) {
        dim3 block(kBN / kTN, kBM / kTM);
        dim3 grid(n / kBN, n / kBM);
        BlockGemmFastKernel<<<grid, block, 0, cache.stream>>>(cache.d_a, cache.d_b, cache.d_c, n);
    } else if (n >= kTile && (n % kTile) == 0) {
        dim3 block(kTile, kTile);
        dim3 grid(n / kTile, n / kTile);
        BlockGemmKernel<<<grid, block, 0, cache.stream>>>(cache.d_a, cache.d_b, cache.d_c, n);
    } else {
        dim3 block(16, 16);
        dim3 grid((n + 15) / 16, (n + 15) / 16);
        BlockGemmSmallKernel<<<grid, block, 0, cache.stream>>>(cache.d_a, cache.d_b, cache.d_c, n);
    }

    std::vector<float> output(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
    cudaMemcpyAsync(cache.h_pin_c, cache.d_c, bytes, cudaMemcpyDeviceToHost, cache.stream);
    cudaStreamSynchronize(cache.stream);
    std::memcpy(output.data(), cache.h_pin_c, bytes);
    return output;
}