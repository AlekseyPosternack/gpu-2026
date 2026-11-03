#include "softmax_cuda.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thread>

#include <cuda_runtime.h>


#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                        \
            std::fprintf(stderr, "CUDA error \"%s\" at %s:%d\n",          \
                         cudaGetErrorString(err_), __FILE__, __LINE__);   \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

namespace {

    constexpr int kWarpSize = 32;

    
    struct MaxOp {
        __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
        __device__ static float Identity() { return -INFINITY; }
    };

    struct SumOp {
        __device__ float operator()(float a, float b) const { return a + b; }
        __device__ static float Identity() { return 0.0f; }
    };
    template <typename Op>
    __device__ float WarpReduce(float value, Op op) {
#pragma unroll
        for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
            value = op(value, __shfl_xor_sync(0xffffffff, value, offset));
        }
        return value;
    }

    template <int kBlock, typename Op>
    __device__ float BlockReduce(float value, Op op) {
        value = WarpReduce(value, op);
        if (kBlock > kWarpSize) {
            constexpr int kWarps = kBlock / kWarpSize;
            __shared__ float warp_results[kWarps];
            __shared__ float block_result;

            const int lane = threadIdx.x % kWarpSize;
            const int warp = threadIdx.x / kWarpSize;
            if (lane == 0) {
                warp_results[warp] = value;
            }
            __syncthreads();
            if (warp == 0) {
                value = (lane < kWarps) ? warp_results[lane] : Op::Identity();
                value = WarpReduce(value, op);
                if (lane == 0) {
                    block_result = value;
                }
            }
            __syncthreads();
            value = block_result;
        }
        return value;
    }

    template <int kBlock, int kItems>
    __global__ void __launch_bounds__(kBlock)
        SoftmaxCachedKernel(float* data, int row_size) {
        float* row = data + static_cast<size_t>(blockIdx.x) * row_size;

        float values[kItems];
        float local_max = -INFINITY;
#pragma unroll
        for (int i = 0; i < kItems; ++i) {
            const int j = threadIdx.x + i * kBlock;
            values[i] = (j < row_size) ? row[j] : -INFINITY;
            local_max = fmaxf(local_max, values[i]);
        }
        const float row_max = BlockReduce<kBlock>(local_max, MaxOp());
        float local_sum = 0.0f;
#pragma unroll
        for (int i = 0; i < kItems; ++i) {
            values[i] = __expf(values[i] - row_max);
            local_sum += values[i];
        }
        const float inv_sum = 1.0f / BlockReduce<kBlock>(local_sum, SumOp());
#pragma unroll
        for (int i = 0; i < kItems; ++i) {
            const int j = threadIdx.x + i * kBlock;
            if (j < row_size) {
                row[j] = values[i] * inv_sum;
            }
        }
    }
    template <int kBlock>
    __global__ void __launch_bounds__(kBlock)
        SoftmaxLongRowKernel(float* data, int row_size) {
        float* row = data + static_cast<size_t>(blockIdx.x) * row_size;

        float local_max = -INFINITY;
        for (int j = threadIdx.x; j < row_size; j += kBlock) {
            local_max = fmaxf(local_max, row[j]);
        }
        const float row_max = BlockReduce<kBlock>(local_max, MaxOp());

        float local_sum = 0.0f;
        for (int j = threadIdx.x; j < row_size; j += kBlock) {
            local_sum += __expf(row[j] - row_max);
        }
        const float inv_sum = 1.0f / BlockReduce<kBlock>(local_sum, SumOp());

        for (int j = threadIdx.x; j < row_size; j += kBlock) {
            row[j] = __expf(row[j] - row_max) * inv_sum;
        }
    }

    template <int kBlock, int kItems>
    void LaunchCached(float* data, int row_count, int row_size) {
        SoftmaxCachedKernel<kBlock, kItems> << <row_count, kBlock >> > (data, row_size);
    }
    void LaunchSoftmax(float* data, int row_count, int row_size) {
        if (row_size <= 32)    LaunchCached<32, 1>(data, row_count, row_size);
        else if (row_size <= 64)    LaunchCached<32, 2>(data, row_count, row_size);
        else if (row_size <= 128)   LaunchCached<32, 4>(data, row_count, row_size);
        else if (row_size <= 256)   LaunchCached<32, 8>(data, row_count, row_size);
        else if (row_size <= 512)   LaunchCached<32, 16>(data, row_count, row_size);
        else if (row_size <= 1024)  LaunchCached<512, 2>(data, row_count, row_size);
        else if (row_size <= 2048)  LaunchCached<512, 4>(data, row_count, row_size);
        else if (row_size <= 4096)  LaunchCached<512, 8>(data, row_count, row_size);
        else if (row_size <= 8192)  LaunchCached<512, 16>(data, row_count, row_size);
        else if (row_size <= 16384) LaunchCached<512, 32>(data, row_count, row_size);
        else SoftmaxLongRowKernel<512> << <row_count, 512 >> > (data, row_size);
    }

    struct DeviceBuffer {
        float* ptr = nullptr;
        size_t capacity = 0;

        float* Get(size_t count) {
            if (count > capacity) {
                if (ptr != nullptr) {
                    CUDA_CHECK(cudaFree(ptr));
                }
                CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(float)));
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

    DeviceBuffer g_device_buffer;

}  // namespace

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (input.empty() || row_count <= 0) {
        return {};
    }
    const size_t count = input.size();
    const int row_size = static_cast<int>(count / row_count);
    const size_t bytes = count * sizeof(float);

    float* d_data = g_device_buffer.Get(count);

    
    std::vector<float> output;
    std::thread alloc_thread([&output, count] { output.resize(count); });

    CUDA_CHECK(cudaMemcpy(d_data, input.data(), bytes, cudaMemcpyHostToDevice));

    LaunchSoftmax(d_data, row_count, row_size);
    CUDA_CHECK(cudaGetLastError());

    alloc_thread.join();
    CUDA_CHECK(cudaMemcpy(output.data(), d_data, bytes, cudaMemcpyDeviceToHost));

    return output;
}