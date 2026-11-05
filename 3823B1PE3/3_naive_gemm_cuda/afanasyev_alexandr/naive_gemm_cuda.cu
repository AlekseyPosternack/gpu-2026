#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace {

__global__ void NaiveGemmKernel(const float* a,
                               const float* b,
                               float* c,
                               int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n || col >= n) {
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
        sum += a[row * n + k] * b[k * n + col];
    }
    c[row * n + col] = sum;
}

}  // namespace

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) {
        return {};
    }

    const size_t matrix_size = static_cast<size_t>(n) * static_cast<size_t>(n);
    if (a.size() != matrix_size || b.size() != matrix_size) {
        throw std::invalid_argument("A and B must both have size n * n");
    }

    const size_t output_size = matrix_size;

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    cudaError_t err = cudaMalloc(&d_a, output_size * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc for A failed: ") + cudaGetErrorString(err));
    }

    err = cudaMalloc(&d_b, output_size * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_a);
        throw std::runtime_error(std::string("cudaMalloc for B failed: ") + cudaGetErrorString(err));
    }

    err = cudaMalloc(&d_c, output_size * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        throw std::runtime_error(std::string("cudaMalloc for C failed: ") + cudaGetErrorString(err));
    }

    err = cudaMemcpy(d_a, a.data(), output_size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        throw std::runtime_error(std::string("cudaMemcpy A failed: ") + cudaGetErrorString(err));
    }

    err = cudaMemcpy(d_b, b.data(), output_size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        throw std::runtime_error(std::string("cudaMemcpy B failed: ") + cudaGetErrorString(err));
    }

    constexpr int block_dim = 16;
    const int grid_x = (n + block_dim - 1) / block_dim;
    const int grid_y = (n + block_dim - 1) / block_dim;
    dim3 threads(block_dim, block_dim);
    dim3 grid(grid_x, grid_y);

    NaiveGemmKernel<<<grid, threads>>>(d_a, d_b, d_c, n);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        throw std::runtime_error(std::string("NaiveGemmKernel launch failed: ") + cudaGetErrorString(err));
    }

    std::vector<float> c(output_size);
    err = cudaMemcpy(c.data(), d_c, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        throw std::runtime_error(std::string("cudaMemcpy C failed: ") + cudaGetErrorString(err));
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}
