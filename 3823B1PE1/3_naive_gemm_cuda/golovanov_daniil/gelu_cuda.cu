#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace {

__global__ void NaiveGemmKernel(const float* a, const float* b, float* c, int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) {
        return;
    }

    float sum = 0.0f;
#pragma unroll 4
    for (int k = 0; k < n; ++k) {
        sum += a[row * n + k] * b[k * n + col];
    }
    c[row * n + col] = sum;
}

void CheckCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

} // namespace

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n < 0 || a.size() != static_cast<std::size_t>(n) * n ||
        b.size() != static_cast<std::size_t>(n) * n) {
        throw std::invalid_argument("matrices must have exactly n*n elements");
    }
    std::vector<float> c(a.size(), 0.0f);
    if (n == 0) {
        return c;
    }

    float *da = nullptr, *db = nullptr, *dc = nullptr;
    const std::size_t bytes = a.size() * sizeof(float);
    CheckCuda(cudaMalloc(&da, bytes), "cudaMalloc(A)");
    try {
        CheckCuda(cudaMalloc(&db, bytes), "cudaMalloc(B)");
        CheckCuda(cudaMalloc(&dc, bytes), "cudaMalloc(C)");
        CheckCuda(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice), "copy A");
        CheckCuda(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice), "copy B");

        const dim3 block(16, 16);
        const dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
        NaiveGemmKernel<<<grid, block>>>(da, db, dc, n);
        CheckCuda(cudaGetLastError(), "NaiveGemmKernel launch");
        CheckCuda(cudaMemcpy(c.data(), dc, bytes, cudaMemcpyDeviceToHost), "copy C");
    } catch (...) {
        cudaFree(dc); cudaFree(db); cudaFree(da);
        throw;
    }
    CheckCuda(cudaFree(dc), "cudaFree(C)");
    CheckCuda(cudaFree(db), "cudaFree(B)");
    CheckCuda(cudaFree(da), "cudaFree(A)");
    return c;
}