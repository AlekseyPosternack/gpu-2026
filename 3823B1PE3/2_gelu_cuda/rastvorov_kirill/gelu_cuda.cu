#include "gelu_cuda.h"

#include <cuda_runtime.h>

namespace {

constexpr float kK0 = -2.0f * 0.7978845608028654f;  // -2 * sqrt(2 / pi)
constexpr float kK1 = kK0 * 0.044715f;

__device__ __forceinline__ float Gelu(float x) {
    // 0.5x(1 + tanh(z)) == x / (1 + exp(-2z))
    return x / (1.0f + __expf(x * (kK0 + kK1 * x * x)));
}

// Works in place (in == out): every element is read and written by one thread
__global__ void GeluKernel(const float* in, float* out, size_t size) {
    const size_t vec_size = size / 4;
    const float4* in4 = reinterpret_cast<const float4*>(in);
    float4* out4 = reinterpret_cast<float4*>(out);
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    for (size_t i = first; i < vec_size; i += stride) {
        float4 v = in4[i];
        v.x = Gelu(v.x);
        v.y = Gelu(v.y);
        v.z = Gelu(v.z);
        v.w = Gelu(v.w);
        out4[i] = v;
    }

    for (size_t i = vec_size * 4 + first; i < size; i += stride) {
        out[i] = Gelu(in[i]);
    }
}

// Device buffer is kept between calls to avoid cudaMalloc/cudaFree each time
float* g_buffer = nullptr;
size_t g_capacity = 0;

void EnsureCapacity(size_t size) {
    if (size <= g_capacity) return;
    if (g_buffer) cudaFree(g_buffer);
    cudaMalloc(&g_buffer, size * sizeof(float));
    g_capacity = size;
}

}  // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const size_t size = input.size();
    if (size == 0) return {};

    EnsureCapacity(size);

    const int block = 256;
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    const size_t max_grid = static_cast<size_t>(sm_count) * 16;
    size_t grid = (size / 4 + block - 1) / block;
    if (grid == 0) grid = 1;
    if (grid > max_grid) grid = max_grid;

    cudaMemcpyAsync(g_buffer, input.data(), size * sizeof(float),
                    cudaMemcpyHostToDevice);
    GeluKernel<<<static_cast<int>(grid), block>>>(g_buffer, g_buffer, size);

    // Host allocation overlaps with the kernel execution
    std::vector<float> output(size);
    cudaMemcpy(output.data(), g_buffer, size * sizeof(float),
               cudaMemcpyDeviceToHost);
    return output;
}
