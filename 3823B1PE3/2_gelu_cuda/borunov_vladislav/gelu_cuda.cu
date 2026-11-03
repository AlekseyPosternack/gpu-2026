#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <cstring>
#include <thread>

namespace {


constexpr float kA = -1.5957691216057308f;    
constexpr float kB = kA * 0.044715f;           

__device__ __forceinline__ float GeluFast(float x) {
    return __fdividef(x, 1.0f + __expf(x * (kA + kB * x * x)));
}

__global__ void GeluKernel(float* __restrict__ data, size_t n4, size_t tail_start, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < n4) {
        float4 v = __ldg(reinterpret_cast<const float4*>(data) + i);
        v.x = GeluFast(v.x);
        v.y = GeluFast(v.y);
        v.z = GeluFast(v.z);
        v.w = GeluFast(v.w);
        reinterpret_cast<float4*>(data)[i] = v;
    } else {
        const size_t tail_i = tail_start + (i - n4);
        if (tail_i < n) {
            data[tail_i] = GeluFast(__ldg(data + tail_i));
        }
    }
}

struct DeviceCache {
    float*  d_buf    = nullptr;
    float*  h_in_pin = nullptr;   
    size_t  d_cap    = 0;
    size_t  h_cap    = 0;
    const float* registered_ptr = nullptr;  

    void ReserveDevice(size_t bytes) {
        if (bytes > d_cap) {
            cudaFree(d_buf);
            cudaMalloc(&d_buf, bytes);
            d_cap = bytes;
        }
    }

    void ReservePinned(size_t bytes) {
        if (bytes > h_cap) {
            cudaFreeHost(h_in_pin);
            cudaMallocHost(&h_in_pin, bytes);
            h_cap = bytes;
        }
    }

    void TryRegister(const float* ptr, size_t bytes) {
        if (ptr == registered_ptr) return;   
        if (registered_ptr) {
            cudaHostUnregister(const_cast<float*>(registered_ptr));
            registered_ptr = nullptr;
        }
        if (cudaHostRegister(const_cast<float*>(ptr), bytes,
                             cudaHostRegisterDefault) == cudaSuccess) {
            registered_ptr = ptr;
        }
    }
};

DeviceCache g_cache;

}  // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const size_t n = input.size();
    if (n == 0) return {};

    const size_t bytes = n * sizeof(float);

    g_cache.ReserveDevice(bytes);

    g_cache.TryRegister(input.data(), bytes);

    std::vector<float> output;
    std::thread alloc_thread([&output, n] { output.resize(n); });

    cudaMemcpy(g_cache.d_buf, input.data(), bytes, cudaMemcpyHostToDevice);

    const size_t n4         = n / 4;
    const size_t tail_start = n4 * 4;
    const size_t work       = n4 + (n % 4);  
    constexpr unsigned kThreads = 256;
    const unsigned blocks = static_cast<unsigned>((work + kThreads - 1) / kThreads);
    GeluKernel<<<blocks, kThreads>>>(g_cache.d_buf, n4, tail_start, n);

    alloc_thread.join();
    cudaMemcpy(output.data(), g_cache.d_buf, bytes, cudaMemcpyDeviceToHost);

    return output;
}

