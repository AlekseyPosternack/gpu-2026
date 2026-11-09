#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <cstddef>

namespace {

__global__ void NaiveGemmFallbackKernel(const float* __restrict__ a,
                                        const float* __restrict__ b,
                                        float* __restrict__ c,
                                        int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

__global__ void NaiveGemmVectorKernel(const float4* __restrict__ a,
                                      const float4* __restrict__ b,
                                      float4* __restrict__ c,
                                      int n) {
    const int n_vec = n / 4;
    const int col_vec = blockIdx.x * blockDim.x + threadIdx.x;
    const int base_row = (blockIdx.y * blockDim.y + threadIdx.y) * 4;

    if (base_row >= n || col_vec >= n_vec) {
        return;
    }

    float4 acc[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        acc[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    const float4* a_row0 = a + (base_row + 0) * n_vec;
    const float4* a_row1 = a + (base_row + 1) * n_vec;
    const float4* a_row2 = a + (base_row + 2) * n_vec;
    const float4* a_row3 = a + (base_row + 3) * n_vec;

    for (int k = 0; k < n_vec; ++k) {
        const float4 va0 = a_row0[k];
        const float4 va1 = a_row1[k];
        const float4 va2 = a_row2[k];
        const float4 va3 = a_row3[k];

        const int k_idx = k * 4;
        const float4 vb0 = b[(k_idx + 0) * n_vec + col_vec];
        const float4 vb1 = b[(k_idx + 1) * n_vec + col_vec];
        const float4 vb2 = b[(k_idx + 2) * n_vec + col_vec];
        const float4 vb3 = b[(k_idx + 3) * n_vec + col_vec];

        #define ACCUMULATE(r, va) \
            acc[r].x = fmaf(va.x, vb0.x, fmaf(va.y, vb1.x, fmaf(va.z, vb2.x, fmaf(va.w, vb3.x, acc[r].x)))); \
            acc[r].y = fmaf(va.x, vb0.y, fmaf(va.y, vb1.y, fmaf(va.z, vb2.y, fmaf(va.w, vb3.y, acc[r].y)))); \
            acc[r].z = fmaf(va.x, vb0.z, fmaf(va.y, vb1.z, fmaf(va.z, vb2.z, fmaf(va.w, vb3.z, acc[r].z)))); \
            acc[r].w = fmaf(va.x, vb0.w, fmaf(va.y, vb1.w, fmaf(va.z, vb2.w, fmaf(va.w, vb3.w, acc[r].w))));

        ACCUMULATE(0, va0);
        ACCUMULATE(1, va1);
        ACCUMULATE(2, va2);
        ACCUMULATE(3, va3);
        #undef ACCUMULATE
    }

    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        c[(base_row + r) * n_vec + col_vec] = acc[r];
    }
}

struct PersistentGpuMemory {
    float* raw_data = nullptr;
    size_t capacity_bytes = 0;

    float* Ensure(size_t required_bytes) {
        if (required_bytes > capacity_bytes) {
            if (raw_data != nullptr) {
                cudaFree(raw_data);
            }
            cudaMalloc(&raw_data, required_bytes);
            capacity_bytes = required_bytes;
        }
        return raw_data;
    }

    ~PersistentGpuMemory() {
        if (raw_data != nullptr) {
            cudaFree(raw_data);
        }
    }
};

PersistentGpuMemory g_memory_pool;

} // namespace

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) {
        return {};
    }

    const size_t total_elements = static_cast<size_t>(n) * n;
    const size_t matrix_bytes = total_elements * sizeof(float);

    float* d_all = g_memory_pool.Ensure(3 * matrix_bytes);
    float* d_a = d_all;
    float* d_b = d_all + total_elements;
    float* d_c = d_all + 2 * total_elements;

    cudaMemcpyAsync(d_a, a.data(), matrix_bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_b, b.data(), matrix_bytes, cudaMemcpyHostToDevice);

    if (n >= 4) {
        dim3 block(16, 16);
        const int n_vec = n / 4;
        dim3 grid((n_vec + block.x - 1) / block.x,
                  (n_vec + block.y - 1) / block.y);

        NaiveGemmVectorKernel<<<grid, block>>>(
            reinterpret_cast<const float4*>(d_a),
            reinterpret_cast<const float4*>(d_b),
            reinterpret_cast<float4*>(d_c),
            n
        );
    } else {
        dim3 block(2, 2);
        dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
        NaiveGemmFallbackKernel<<<grid, block>>>(d_a, d_b, d_c, n);
    }

    std::vector<float> c(total_elements);

    cudaMemcpy(c.data(), d_c, matrix_bytes, cudaMemcpyDeviceToHost);

    return c;
}