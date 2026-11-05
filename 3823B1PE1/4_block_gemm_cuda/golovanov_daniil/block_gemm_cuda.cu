#include "block_gemm_cuda.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace {

constexpr int kTile = 32;

__global__ void block_gemm_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c,
                                  int n) {
    __shared__ float As[kTile][kTile];
    __shared__ float Bs[kTile][kTile];
    __shared__ float Cs[kTile][kTile];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * kTile + ty;
    const int col = blockIdx.x * kTile + tx;

    float acc = 0.0f;

    for (int t = 0; t < n; t += kTile) {
        if (row < n && (t + tx) < n)
            As[ty][tx] = a[static_cast<std::size_t>(row) * n + t + tx];
        else
            As[ty][tx] = 0.0f;

        if ((t + ty) < n && col < n)
            Bs[ty][tx] = b[static_cast<std::size_t>(t + ty) * n + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < kTile; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    Cs[ty][tx] = acc;
    __syncthreads();

    if (row < n && col < n)
        c[static_cast<std::size_t>(row) * n + col] = Cs[ty][tx];
}

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

struct DeviceBuffer {
    float* ptr = nullptr;
    std::size_t cap = 0;

    void ensure(std::size_t n) {
        if (n > cap) {
            if (ptr) check_cuda(cudaFree(ptr), "cudaFree");
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(float)),
                       "cudaMalloc");
            cap = n;
        }
    }

    ~DeviceBuffer() {
        if (ptr) cudaFree(ptr);
    }
};

} // namespace

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    const std::size_t total = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    std::vector<float> c(total);
    if (total == 0) return c;

    static DeviceBuffer d_a;
    static DeviceBuffer d_b;
    static DeviceBuffer d_c;

    d_a.ensure(total);
    d_b.ensure(total);
    d_c.ensure(total);

    const std::size_t bytes = total * sizeof(float);

    check_cuda(cudaMemcpyAsync(d_a.ptr, a.data(), bytes, cudaMemcpyHostToDevice, 0),
               "cudaMemcpy a H2D");
    check_cuda(cudaMemcpyAsync(d_b.ptr, b.data(), bytes, cudaMemcpyHostToDevice, 0),
               "cudaMemcpy b H2D");

    const dim3 block(kTile, kTile);
    const dim3 grid((n + kTile - 1) / kTile, (n + kTile - 1) / kTile);

    block_gemm_kernel<<<grid, block, 0, 0>>>(d_a.ptr, d_b.ptr, d_c.ptr, n);

    check_cuda(cudaGetLastError(), "block_gemm_kernel launch");

    check_cuda(cudaMemcpyAsync(c.data(), d_c.ptr, bytes, cudaMemcpyDeviceToHost, 0),
               "cudaMemcpy c D2H");

    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    return c;
}