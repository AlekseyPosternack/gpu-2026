#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>

// Fallback kernel for edge-case small power-of-2 matrices (n = 1 or n = 2)
__global__ void naive_gemm_small_kernel(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C,
                                        int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Optimized naive kernel for n >= 4 (always a power of 2, so n is a multiple of 4).
// Each thread computes a 4x4 block of C using float4 vectorized loads and stores.
__global__ void naive_gemm_vec4_kernel(const float4* __restrict__ A,
                                       const float4* __restrict__ B,
                                       float4* __restrict__ C,
                                       int n) {
    // col4 represents the index in terms of float4 elements (each is 4 floats wide)
    int col4 = blockIdx.x * blockDim.x + threadIdx.x;
    int row = (blockIdx.y * blockDim.y + threadIdx.y) * 4;

    int n4 = n / 4;
    if (row >= n || col4 >= n4) return;

    // 16 registers to accumulate a 4x4 tile of C per thread
    float4 c0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 c1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 c2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 c3 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    // Each row of A has n4 float4 elements
    const float4* A0 = A + row * n4;
    const float4* A1 = A + (row + 1) * n4;
    const float4* A2 = A + (row + 2) * n4;
    const float4* A3 = A + (row + 3) * n4;

    for (int k4 = 0; k4 < n4; ++k4) {
        // Load 4 horizontal elements (k, k+1, k+2, k+3) for each of the 4 rows of A
        float4 a0 = __ldg(&A0[k4]);
        float4 a1 = __ldg(&A1[k4]);
        float4 a2 = __ldg(&A2[k4]);
        float4 a3 = __ldg(&A3[k4]);

        // Load the 4 corresponding rows of B at column group col4
        int k_base = k4 * 4;
        float4 b0 = __ldg(&B[(k_base + 0) * n4 + col4]);
        float4 b1 = __ldg(&B[(k_base + 1) * n4 + col4]);
        float4 b2 = __ldg(&B[(k_base + 2) * n4 + col4]);
        float4 b3 = __ldg(&B[(k_base + 3) * n4 + col4]);

        // Accumulate k + 0
        c0.x += a0.x * b0.x; c0.y += a0.x * b0.y; c0.z += a0.x * b0.z; c0.w += a0.x * b0.w;
        c1.x += a1.x * b0.x; c1.y += a1.x * b0.y; c1.z += a1.x * b0.z; c1.w += a1.x * b0.w;
        c2.x += a2.x * b0.x; c2.y += a2.x * b0.y; c2.z += a2.x * b0.z; c2.w += a2.x * b0.w;
        c3.x += a3.x * b0.x; c3.y += a3.x * b0.y; c3.z += a3.x * b0.z; c3.w += a3.x * b0.w;

        // Accumulate k + 1
        c0.x += a0.y * b1.x; c0.y += a0.y * b1.y; c0.z += a0.y * b1.z; c0.w += a0.y * b1.w;
        c1.x += a1.y * b1.x; c1.y += a1.y * b1.y; c1.z += a1.y * b1.z; c1.w += a1.y * b1.w;
        c2.x += a2.y * b1.x; c2.y += a2.y * b1.y; c2.z += a2.y * b1.z; c2.w += a2.y * b1.w;
        c3.x += a3.y * b1.x; c3.y += a3.y * b1.y; c3.z += a3.y * b1.z; c3.w += a3.y * b1.w;

        // Accumulate k + 2
        c0.x += a0.z * b2.x; c0.y += a0.z * b2.y; c0.z += a0.z * b2.z; c0.w += a0.z * b2.w;
        c1.x += a1.z * b2.x; c1.y += a1.z * b2.y; c1.z += a1.z * b2.z; c1.w += a1.z * b2.w;
        c2.x += a2.z * b2.x; c2.y += a2.z * b2.y; c2.z += a2.z * b2.z; c2.w += a2.z * b2.w;
        c3.x += a3.z * b2.x; c3.y += a3.z * b2.y; c3.z += a3.z * b2.z; c3.w += a3.z * b2.w;

        // Accumulate k + 3
        c0.x += a0.w * b3.x; c0.y += a0.w * b3.y; c0.z += a0.w * b3.z; c0.w += a0.w * b3.w;
        c1.x += a1.w * b3.x; c1.y += a1.w * b3.y; c1.z += a1.w * b3.z; c1.w += a1.w * b3.w;
        c2.x += a2.w * b3.x; c2.y += a2.w * b3.y; c2.z += a2.w * b3.z; c2.w += a2.w * b3.w;
        c3.x += a3.w * b3.x; c3.y += a3.w * b3.y; c3.z += a3.w * b3.z; c3.w += a3.w * b3.w;
    }

    // Coalesced 128-bit vectorized stores to C
    C[(row + 0) * n4 + col4] = c0;
    C[(row + 1) * n4 + col4] = c1;
    C[(row + 2) * n4 + col4] = c2;
    C[(row + 3) * n4 + col4] = c3;
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) return {};

    size_t num_elements = static_cast<size_t>(n) * n;
    size_t bytes = num_elements * sizeof(float);

    // Allocate a single contiguous device memory block for A, B, and C
    // Since bytes is a multiple of 4 (and 16 for n >= 2), float4 alignment is guaranteed.
    float* d_mem = nullptr;
    cudaMalloc(&d_mem, 3 * bytes);

    float* d_A = d_mem;
    float* d_B = d_mem + num_elements;
    float* d_C = d_mem + 2 * num_elements;

    // Asynchronously copy inputs A and B to the device
    cudaMemcpyAsync(d_A, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_B, b.data(), bytes, cudaMemcpyHostToDevice);

    // Select block size and launch the appropriate kernel
    if (n >= 4) {
        // 16x16 = 256 threads per block.
        // Each block computes a 64x64 tile of C (16*4 rows by 16*4 columns).
        dim3 block(16, 16);
        int n4 = n / 4;
        dim3 grid((n4 + block.x - 1) / block.x,
                  (n4 + block.y - 1) / block.y);

        naive_gemm_vec4_kernel<<<grid, block>>>(
            reinterpret_cast<const float4*>(d_A),
            reinterpret_cast<const float4*>(d_B),
            reinterpret_cast<float4*>(d_C),
            n);
    } else {
        dim3 block(2, 2);
        dim3 grid((n + block.x - 1) / block.x,
                  (n + block.y - 1) / block.y);
        naive_gemm_small_kernel<<<grid, block>>>(d_A, d_B, d_C, n);
    }

    // Overlap host vector allocation and zero-initialization with GPU execution
    std::vector<float> c(num_elements);

    // Copy result back to host (implicitly synchronizes with the default stream)
    cudaMemcpy(c.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_mem);

    return c;
}