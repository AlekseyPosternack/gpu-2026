#include "gelu_cuda.h"

#include <cuda_runtime.h>

namespace {

constexpr int kBlockSize = 256;

__global__ void GeluKernel(const float4* __restrict__ in, float4* __restrict__ out,
                           const float* __restrict__ in_tail, float* __restrict__ out_tail,
                           size_t vec_count, size_t tail_count) {
    const float k1 = -2.0f * 0.7978845608028654f;
    const float k2 = k1 * 0.044715f;

    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < vec_count; i += stride) {
        float4 v = in[i];
        v.x = __fdividef(v.x, 1.0f + __expf(v.x * (k1 + k2 * v.x * v.x)));
        v.y = __fdividef(v.y, 1.0f + __expf(v.y * (k1 + k2 * v.y * v.y)));
        v.z = __fdividef(v.z, 1.0f + __expf(v.z * (k1 + k2 * v.z * v.z)));
        v.w = __fdividef(v.w, 1.0f + __expf(v.w * (k1 + k2 * v.w * v.w)));
        out[i] = v;
    }

    if (blockIdx.x == 0 && threadIdx.x < tail_count) {
        const float x = in_tail[threadIdx.x];
        out_tail[threadIdx.x] = __fdividef(x, 1.0f + __expf(x * (k1 + k2 * x * x)));
    }
}

struct DeviceBuffers {
    float* in = nullptr;
    float* out = nullptr;
    size_t capacity = 0;
    cudaStream_t stream = nullptr;

    void Reserve(size_t size) {
        if (!stream) {
            cudaStreamCreate(&stream);
        }
        if (size <= capacity) {
            return;
        }
        cudaFree(in);
        cudaFree(out);
        cudaMalloc(&in, size * sizeof(float));
        cudaMalloc(&out, size * sizeof(float));
        capacity = size;
    }

    ~DeviceBuffers() {
        cudaFree(in);
        cudaFree(out);
        if (stream) {
            cudaStreamDestroy(stream);
        }
    }
};

DeviceBuffers buffers;

}  // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const size_t size = input.size();
    if (size == 0) {
        return {};
    }
    buffers.Reserve(size);
    const size_t bytes = size * sizeof(float);

    cudaMemcpyAsync(buffers.in, input.data(), bytes, cudaMemcpyHostToDevice, buffers.stream);

    const size_t vec_count = size / 4;
    const size_t tail_count = size % 4;
    int device = 0;
    int sm_count = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    size_t grid = (vec_count + kBlockSize - 1) / kBlockSize;
    const size_t max_grid = static_cast<size_t>(sm_count) * 32;
    if (grid > max_grid) grid = max_grid;
    if (grid == 0) grid = 1;

    GeluKernel<<<static_cast<unsigned>(grid), kBlockSize, 0, buffers.stream>>>(
        reinterpret_cast<const float4*>(buffers.in), reinterpret_cast<float4*>(buffers.out),
        buffers.in + vec_count * 4, buffers.out + vec_count * 4, vec_count, tail_count);

    std::vector<float> output(size);
    cudaMemcpyAsync(output.data(), buffers.out, bytes, cudaMemcpyDeviceToHost, buffers.stream);
    cudaStreamSynchronize(buffers.stream);
    return output;
}
