#include "gelu_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

namespace {

constexpr float kK1 = -1.5957691216057308f;
constexpr float kK2 = -0.0713548162726009f;
constexpr int kBlockSize = 256;

__device__ __forceinline__ float GeluFast(float x) {
    return __fdividef(x, 1.0f + __expf(x * (kK1 + kK2 * x * x)));
}

__global__ void GeluKernel(float* __restrict__ data, size_t n) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t n4 = n / 4;

    if (idx < n4) {
        float4* data4 = reinterpret_cast<float4*>(data);
        float4 v = data4[idx];
        v.x = GeluFast(v.x);
        v.y = GeluFast(v.y);
        v.z = GeluFast(v.z);
        v.w = GeluFast(v.w);
        data4[idx] = v;
    } else {
        const size_t tail_idx = idx - n4;
        const size_t rem = n % 4;
        if (tail_idx < rem) {
            const size_t scalar_idx = n4 * 4 + tail_idx;
            data[scalar_idx] = GeluFast(data[scalar_idx]);
        }
    }
}

struct DeviceBuffer {
    float* data = nullptr;
    size_t capacity = 0;
    cudaStream_t stream = nullptr;

    DeviceBuffer() {
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    }

    ~DeviceBuffer() {
        if (data) {
            cudaFree(data);
        }
        if (stream) {
            cudaStreamDestroy(stream);
        }
    }

    float* Get(size_t n) {
        if (n > capacity) {
            if (data) {
                cudaFree(data);
            }
            cudaMalloc(&data, n * sizeof(float));
            capacity = n;
        }
        return data;
    }
};

DeviceBuffer& GetDeviceBuffer() {
    static DeviceBuffer buffer;
    return buffer;
}

}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const size_t n = input.size();
    if (n == 0) {
        return {};
    }

    const size_t bytes = n * sizeof(float);
    DeviceBuffer& buf = GetDeviceBuffer();
    float* d_data = buf.Get(n);

    std::vector<float> output;
    std::thread allocator([&output, n] {
        output.resize(n);
    });

    const bool in_pinned = (cudaHostRegister(const_cast<float*>(input.data()), bytes, cudaHostRegisterDefault) == cudaSuccess);

    cudaMemcpyAsync(d_data, input.data(), bytes, cudaMemcpyHostToDevice, buf.stream);

    const size_t n4 = n / 4;
    const size_t rem = n % 4;
    const size_t total_threads = n4 + rem;
    const unsigned int blocks = static_cast<unsigned int>((total_threads + kBlockSize - 1) / kBlockSize);

    GeluKernel<<<blocks, kBlockSize, 0, buf.stream>>>(d_data, n);

    allocator.join();

    const bool out_pinned = (cudaHostRegister(output.data(), bytes, cudaHostRegisterDefault) == cudaSuccess);

    cudaMemcpyAsync(output.data(), d_data, bytes, cudaMemcpyDeviceToHost, buf.stream);
    cudaStreamSynchronize(buf.stream);

    if (in_pinned) {
        cudaHostUnregister(const_cast<float*>(input.data()));
    }
    if (out_pinned) {
        cudaHostUnregister(output.data());
    }

    return output;
}
