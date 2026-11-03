#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <thread>

namespace {

__global__ void NaiveGemmSmallKernel(const float* a, const float* b, float* c, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

__global__ void NaiveGemmKernel(const float4* __restrict__ a,
                                const float4* __restrict__ b,
                                float4* __restrict__ c,
                                int n) {
    int col4 = blockIdx.x * blockDim.x + threadIdx.x;
    int row = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
    int n4 = n / 4;

    if (row >= n || col4 >= n4) {
        return;
    }

    float4 c0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 c1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 c2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 c3 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    const float4* a0 = a + row * n4;
    const float4* a1 = a + (row + 1) * n4;
    const float4* a2 = a + (row + 2) * n4;
    const float4* a3 = a + (row + 3) * n4;

    for (int k4 = 0; k4 < n4; ++k4) {
        float4 a0_val = a0[k4];
        float4 a1_val = a1[k4];
        float4 a2_val = a2[k4];
        float4 a3_val = a3[k4];

        int k_base = k4 * 4;
        float4 b0 = b[(k_base + 0) * n4 + col4];
        float4 b1 = b[(k_base + 1) * n4 + col4];
        float4 b2 = b[(k_base + 2) * n4 + col4];
        float4 b3 = b[(k_base + 3) * n4 + col4];

        c0.x += a0_val.x * b0.x + a0_val.y * b1.x + a0_val.z * b2.x + a0_val.w * b3.x;
        c0.y += a0_val.x * b0.y + a0_val.y * b1.y + a0_val.z * b2.y + a0_val.w * b3.y;
        c0.z += a0_val.x * b0.z + a0_val.y * b1.z + a0_val.z * b2.z + a0_val.w * b3.z;
        c0.w += a0_val.x * b0.w + a0_val.y * b1.w + a0_val.z * b2.w + a0_val.w * b3.w;

        c1.x += a1_val.x * b0.x + a1_val.y * b1.x + a1_val.z * b2.x + a1_val.w * b3.x;
        c1.y += a1_val.x * b0.y + a1_val.y * b1.y + a1_val.z * b2.y + a1_val.w * b3.y;
        c1.z += a1_val.x * b0.z + a1_val.y * b1.z + a1_val.z * b2.z + a1_val.w * b3.z;
        c1.w += a1_val.x * b0.w + a1_val.y * b1.w + a1_val.z * b2.w + a1_val.w * b3.w;

        c2.x += a2_val.x * b0.x + a2_val.y * b1.x + a2_val.z * b2.x + a2_val.w * b3.x;
        c2.y += a2_val.x * b0.y + a2_val.y * b1.y + a2_val.z * b2.y + a2_val.w * b3.y;
        c2.z += a2_val.x * b0.z + a2_val.y * b1.z + a2_val.z * b2.z + a2_val.w * b3.z;
        c2.w += a2_val.x * b0.w + a2_val.y * b1.w + a2_val.z * b2.w + a2_val.w * b3.w;

        c3.x += a3_val.x * b0.x + a3_val.y * b1.x + a3_val.z * b2.x + a3_val.w * b3.x;
        c3.y += a3_val.x * b0.y + a3_val.y * b1.y + a3_val.z * b2.y + a3_val.w * b3.y;
        c3.z += a3_val.x * b0.z + a3_val.y * b1.z + a3_val.z * b2.z + a3_val.w * b3.z;
        c3.w += a3_val.x * b0.w + a3_val.y * b1.w + a3_val.z * b2.w + a3_val.w * b3.w;
    }

    c[(row + 0) * n4 + col4] = c0;
    c[(row + 1) * n4 + col4] = c1;
    c[(row + 2) * n4 + col4] = c2;
    c[(row + 3) * n4 + col4] = c3;
}

struct DeviceBuffer {
    float* ptr = nullptr;
    size_t capacity = 0;

    float* Get(size_t count) {
        if (count > capacity) {
            if (ptr != nullptr) {
                cudaFree(ptr);
            }
            cudaMalloc(&ptr, count * sizeof(float));
            capacity = count;
        }
        return ptr;
    }

    ~DeviceBuffer() {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }
};

DeviceBuffer g_device_mem;

}  // namespace

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) {
        return {};
    }

    size_t num_elements = static_cast<size_t>(n) * n;
    size_t bytes = num_elements * sizeof(float);

    float* d_mem = g_device_mem.Get(3 * num_elements);
    float* d_a = d_mem;
    float* d_b = d_mem + num_elements;
    float* d_c = d_mem + 2 * num_elements;

    std::vector<float> c;
    std::thread alloc_thread([&c, num_elements] { c.resize(num_elements); });

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    if (n >= 4) {
        dim3 block(16, 16);
        int n4 = n / 4;
        dim3 grid((n4 + block.x - 1) / block.x, (n4 + block.y - 1) / block.y);
        NaiveGemmKernel<<<grid, block>>>(reinterpret_cast<const float4*>(d_a),
                                         reinterpret_cast<const float4*>(d_b),
                                         reinterpret_cast<float4*>(d_c),
                                         n);
    } else {
        dim3 block(2, 2);
        dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
        NaiveGemmSmallKernel<<<grid, block>>>(d_a, d_b, d_c, n);
    }

    alloc_thread.join();

    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    return c;
}
