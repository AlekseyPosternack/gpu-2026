#include "block_gemm_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <thread>
#include <vector>

namespace {

constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileK = 16;
constexpr int kThreadM = 4;
constexpr int kThreadN = 4;
constexpr int kPadA = 1;

__device__ __forceinline__ float4 LoadFloat4(const float* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

__device__ __forceinline__ void StoreFloat4(float* ptr, float4 val) {
    *reinterpret_cast<float4*>(ptr) = val;
}

__global__ void __launch_bounds__(256) BlockGemmKernel(const float* __restrict__ a,
                                                        const float* __restrict__ b,
                                                        float* __restrict__ c,
                                                        int n) {
    __shared__ float s_a[kTileM][kTileK + kPadA];
    __shared__ float s_b[kTileK][kTileN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    const int load_a_row = tid / (kTileK / 4);
    const int load_a_col = (tid % (kTileK / 4)) * 4;

    const int load_b_row = tid / (kTileN / 4);
    const int load_b_col = (tid % (kTileN / 4)) * 4;

    const int block_row = blockIdx.y * kTileM;
    const int block_col = blockIdx.x * kTileN;

    float acc[kThreadM][kThreadN];
    #pragma unroll
    for (int i = 0; i < kThreadM; ++i) {
        #pragma unroll
        for (int j = 0; j < kThreadN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int k_step = 0; k_step < n; k_step += kTileK) {
        const float* global_a = a + (block_row + load_a_row) * n + (k_step + load_a_col);
        const float4 a_val = LoadFloat4(global_a);
        s_a[load_a_row][load_a_col + 0] = a_val.x;
        s_a[load_a_row][load_a_col + 1] = a_val.y;
        s_a[load_a_row][load_a_col + 2] = a_val.z;
        s_a[load_a_row][load_a_col + 3] = a_val.w;

        const float* global_b = b + (k_step + load_b_row) * n + (block_col + load_b_col);
        const float4 b_val = LoadFloat4(global_b);
        StoreFloat4(&s_b[load_b_row][load_b_col], b_val);

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < kTileK; ++k) {
            float a_reg[kThreadM];
            #pragma unroll
            for (int i = 0; i < kThreadM; ++i) {
                a_reg[i] = s_a[ty * kThreadM + i][k];
            }

            const float4 b_val_k = LoadFloat4(&s_b[k][tx * kThreadN]);

            #pragma unroll
            for (int i = 0; i < kThreadM; ++i) {
                acc[i][0] = fmaf(a_reg[i], b_val_k.x, acc[i][0]);
                acc[i][1] = fmaf(a_reg[i], b_val_k.y, acc[i][1]);
                acc[i][2] = fmaf(a_reg[i], b_val_k.z, acc[i][2]);
                acc[i][3] = fmaf(a_reg[i], b_val_k.w, acc[i][3]);
            }
        }

        __syncthreads();
    }

    const int out_row_base = block_row + ty * kThreadM;
    const int out_col_base = block_col + tx * kThreadN;

    #pragma unroll
    for (int i = 0; i < kThreadM; ++i) {
        float* out_ptr = c + (out_row_base + i) * n + out_col_base;
        StoreFloat4(out_ptr, make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]));
    }
}

constexpr int kSmallTile = 16;
__global__ void BlockGemmSmallKernel(const float* __restrict__ a,
                                     const float* __restrict__ b,
                                     float* __restrict__ c,
                                     int n) {
    __shared__ float s_a[kSmallTile][kSmallTile];
    __shared__ float s_b[kSmallTile][kSmallTile];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * kSmallTile + ty;
    const int col = blockIdx.x * kSmallTile + tx;

    float sum = 0.0f;
    for (int k_step = 0; k_step < n; k_step += kSmallTile) {
        s_a[ty][tx] = (row < n && (k_step + tx) < n) ? a[row * n + k_step + tx] : 0.0f;
        s_b[ty][tx] = ((k_step + ty) < n && col < n) ? b[(k_step + ty) * n + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < kSmallTile; ++k) {
            sum = fmaf(s_a[ty][k], s_b[k][tx], sum);
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

struct DeviceMemoryManager {
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    size_t capacity = 0;
    cudaStream_t stream = nullptr;

    DeviceMemoryManager() {
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    }

    ~DeviceMemoryManager() {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
        if (stream) cudaStreamDestroy(stream);
    }

    void EnsureCapacity(size_t elements) {
        if (elements > capacity) {
            if (d_a) cudaFree(d_a);
            if (d_b) cudaFree(d_b);
            if (d_c) cudaFree(d_c);
            cudaMalloc(&d_a, elements * sizeof(float));
            cudaMalloc(&d_b, elements * sizeof(float));
            cudaMalloc(&d_c, elements * sizeof(float));
            capacity = elements;
        }
    }
};

DeviceMemoryManager& GetMemoryManager() {
    static DeviceMemoryManager manager;
    return manager;
}

} // namespace

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) {
        return {};
    }

    const size_t total_elements = static_cast<size_t>(n) * static_cast<size_t>(n);
    const size_t bytes = total_elements * sizeof(float);

    DeviceMemoryManager& mem = GetMemoryManager();
    mem.EnsureCapacity(total_elements);

    std::vector<float> output;
    std::thread host_allocator([&output, total_elements]() {
        output.resize(total_elements);
    });

    const bool a_pinned = (cudaHostRegister(const_cast<float*>(a.data()), bytes, cudaHostRegisterDefault) == cudaSuccess);
    const bool b_pinned = (cudaHostRegister(const_cast<float*>(b.data()), bytes, cudaHostRegisterDefault) == cudaSuccess);

    cudaMemcpyAsync(mem.d_a, a.data(), bytes, cudaMemcpyHostToDevice, mem.stream);
    cudaMemcpyAsync(mem.d_b, b.data(), bytes, cudaMemcpyHostToDevice, mem.stream);

    if (n >= kTileM && (n % kTileM == 0) && (n % kTileN == 0)) {
        dim3 block(kTileN / kThreadN, kTileM / kThreadM);
        dim3 grid(n / kTileN, n / kTileM);
        BlockGemmKernel<<<grid, block, 0, mem.stream>>>(mem.d_a, mem.d_b, mem.d_c, n);
    } else {
        dim3 block(kSmallTile, kSmallTile);
        dim3 grid((n + kSmallTile - 1) / kSmallTile, (n + kSmallTile - 1) / kSmallTile);
        BlockGemmSmallKernel<<<grid, block, 0, mem.stream>>>(mem.d_a, mem.d_b, mem.d_c, n);
    }

    host_allocator.join();

    const bool out_pinned = (cudaHostRegister(output.data(), bytes, cudaHostRegisterDefault) == cudaSuccess);

    cudaMemcpyAsync(output.data(), mem.d_c, bytes, cudaMemcpyDeviceToHost, mem.stream);
    cudaStreamSynchronize(mem.stream);

    if (a_pinned) cudaHostUnregister(const_cast<float*>(a.data()));
    if (b_pinned) cudaHostUnregister(const_cast<float*>(b.data()));
    if (out_pinned) cudaHostUnregister(output.data());

    return output;
}
