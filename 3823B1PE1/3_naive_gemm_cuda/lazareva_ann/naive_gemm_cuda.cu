#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err__ = (call);                                        \
        if (err__ != cudaSuccess) {                                        \
            throw std::runtime_error(std::string("CUDA error: ") +         \
                                      cudaGetErrorString(err__));           \
        }                                                                  \                                                             
    } while (0)

namespace {

constexpr int kBlockDimX = 16;  
constexpr int kBlockDimY = 16;  
constexpr int kTileX     = 4;   

__global__ void NaiveGemmKernelVec4(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int n) {
    int row     = blockIdx.y * blockDim.y + threadIdx.y;
    int colBase = (blockIdx.x * blockDim.x + threadIdx.x) * kTileX;

    if (row >= n || colBase >= n) return;

    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;

    const float* aRow = A + (size_t)row * n;
    const float* bCol = B + colBase;

    #pragma unroll 4
    for (int k = 0; k < n; ++k) {
        float  aVal = __ldg(&aRow[k]);
        float4 bVal = *reinterpret_cast<const float4*>(bCol + (size_t)k * n);
        acc0 += aVal * bVal.x;
        acc1 += aVal * bVal.y;
        acc2 += aVal * bVal.z;
        acc3 += aVal * bVal.w;
    }

    float4 result = make_float4(acc0, acc1, acc2, acc3);
    *reinterpret_cast<float4*>(C + (size_t)row * n + colBase) = result;
}

__global__ void NaiveGemmKernelScalar(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) return;

    float acc = 0.f;
    for (int k = 0; k < n; ++k) {
        acc += A[row * n + k] * B[k * n + col];
    }
    C[row * n + col] = acc;
}

struct CudaResources {
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    size_t capacityElems = 0;
    cudaStream_t stream = nullptr;
    bool ready = false;

    void EnsureCapacity(size_t elems) {
        if (!ready) {
            CUDA_CHECK(cudaStreamCreate(&stream));
            ready = true;
        }
        if (elems > capacityElems) {
            if (d_a) CUDA_CHECK(cudaFree(d_a));
            if (d_b) CUDA_CHECK(cudaFree(d_b));
            if (d_c) CUDA_CHECK(cudaFree(d_c));
            CUDA_CHECK(cudaMalloc(&d_a, elems * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_b, elems * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_c, elems * sizeof(float)));
            capacityElems = elems;
        }
    }

    ~CudaResources() {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
        if (ready) cudaStreamDestroy(stream);
    }
};

CudaResources g_res;

}  // namespace

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                  const std::vector<float>& b,
                                  int n) {
    const size_t elems = (size_t)n * (size_t)n;
    std::vector<float> c(elems);
    if (n == 0) return c;

    g_res.EnsureCapacity(elems);
    cudaStream_t stream = g_res.stream;

    float* h_a = const_cast<float*>(a.data());
    float* h_b = const_cast<float*>(b.data());
    float* h_c = c.data();

    cudaError_t regA = cudaHostRegister(h_a, elems * sizeof(float), cudaHostRegisterDefault);
    cudaError_t regB = cudaHostRegister(h_b, elems * sizeof(float), cudaHostRegisterDefault);
    cudaError_t regC = cudaHostRegister(h_c, elems * sizeof(float), cudaHostRegisterDefault);

    CUDA_CHECK(cudaMemcpyAsync(g_res.d_a, h_a, elems * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(g_res.d_b, h_b, elems * sizeof(float),
                               cudaMemcpyHostToDevice, stream));

    if (n >= kTileX && n % kTileX == 0) {
        dim3 block(kBlockDimX, kBlockDimY);
        dim3 grid((n / kTileX + block.x - 1) / block.x,
                  (n + block.y - 1) / block.y);
        NaiveGemmKernelVec4<<<grid, block, 0, stream>>>(g_res.d_a, g_res.d_b, g_res.d_c, n);
    } else {
        dim3 block(16, 16);
        dim3 grid((n + block.x - 1) / block.x,
                  (n + block.y - 1) / block.y);
        NaiveGemmKernelScalar<<<grid, block, 0, stream>>>(g_res.d_a, g_res.d_b, g_res.d_c, n);
    }

    CUDA_CHECK(cudaMemcpyAsync(h_c, g_res.d_c, elems * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (regA == cudaSuccess) cudaHostUnregister(h_a);
    if (regB == cudaSuccess) cudaHostUnregister(h_b);
    if (regC == cudaSuccess) cudaHostUnregister(h_c);

    return c;
}