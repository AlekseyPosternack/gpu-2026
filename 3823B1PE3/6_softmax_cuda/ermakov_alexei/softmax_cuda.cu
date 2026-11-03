#include "softmax_cuda.h"

#include <cuda_runtime.h>
#include <math_constants.h>
#include <cub/block/block_reduce.cuh>

#include <cstddef>
#include <future>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

constexpr int threads = 256;

void Check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

struct Maximum {
    __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
};

__global__ void ShortRows(float* data, std::size_t columns, unsigned rows) {
    const unsigned row = blockIdx.x * 4 + threadIdx.x / 32;
    if (row >= rows) return;
    const unsigned lane = threadIdx.x % 32;
    float* values = data + static_cast<std::size_t>(row) * columns;
    float cached[4];
    float maximum = -CUDART_INF_F;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        cached[i] = -CUDART_INF_F;
        if (lane + i * 32 < columns) cached[i] = values[lane + i * 32];
        maximum = fmaxf(maximum, cached[i]);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, offset));
    }
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        cached[i] = __expf(cached[i] - maximum);
        sum += cached[i];
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    }
    const float scale = __fdividef(1.0f, sum);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (lane + i * 32 < columns) values[lane + i * 32] = cached[i] * scale;
    }
}

template<int Vectors>
__global__ void CachedRows(float* data, std::size_t columns) {
    using Reduction = cub::BlockReduce<float, threads>;
    __shared__ typename Reduction::TempStorage temporary;
    __shared__ float statistic;
    float4* row = reinterpret_cast<float4*>(data + static_cast<std::size_t>(blockIdx.x) * columns);
    const std::size_t count = columns / 4;
    float4 cached[Vectors];
    float maximum = -CUDART_INF_F;
#pragma unroll
    for (int i = 0; i < Vectors; ++i) {
        cached[i] = make_float4(-CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F);
        const std::size_t index = threadIdx.x + i * threads;
        if (index < count) cached[i] = row[index];
        maximum = fmaxf(maximum, fmaxf(fmaxf(cached[i].x, cached[i].y),
                                      fmaxf(cached[i].z, cached[i].w)));
    }
    maximum = Reduction(temporary).Reduce(maximum, Maximum{});
    if (threadIdx.x == 0) statistic = maximum;
    __syncthreads();
    maximum = statistic;
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < Vectors; ++i) {
        cached[i].x = __expf(cached[i].x - maximum);
        cached[i].y = __expf(cached[i].y - maximum);
        cached[i].z = __expf(cached[i].z - maximum);
        cached[i].w = __expf(cached[i].w - maximum);
        sum += (cached[i].x + cached[i].y) + (cached[i].z + cached[i].w);
    }
    sum = Reduction(temporary).Sum(sum);
    if (threadIdx.x == 0) statistic = __fdividef(1.0f, sum);
    __syncthreads();
    const float scale = statistic;
#pragma unroll
    for (int i = 0; i < Vectors; ++i) {
        const std::size_t index = threadIdx.x + i * threads;
        if (index < count) {
            row[index] = make_float4(cached[i].x * scale, cached[i].y * scale,
                                     cached[i].z * scale, cached[i].w * scale);
        }
    }
}

__global__ void LongRows(float* data, std::size_t columns) {
    using Reduction = cub::BlockReduce<float, threads>;
    using SumReduction = cub::BlockReduce<double, threads>;
    __shared__ typename Reduction::TempStorage temporary;
    __shared__ typename SumReduction::TempStorage sum_temporary;
    __shared__ float statistic;
    float* row = data + static_cast<std::size_t>(blockIdx.x) * columns;
    float maximum = -CUDART_INF_F;
    for (std::size_t i = threadIdx.x; i < columns; i += threads) {
        maximum = fmaxf(maximum, row[i]);
    }
    maximum = Reduction(temporary).Reduce(maximum, Maximum{});
    if (threadIdx.x == 0) statistic = maximum;
    __syncthreads();
    maximum = statistic;
    double sum = 0.0;
    for (std::size_t i = threadIdx.x; i < columns; i += threads) {
        sum += __expf(row[i] - maximum);
    }
    sum = SumReduction(sum_temporary).Sum(sum);
    if (threadIdx.x == 0) statistic = static_cast<float>(1.0 / sum);
    __syncthreads();
    const float scale = statistic;
    for (std::size_t i = threadIdx.x; i < columns; i += threads) {
        row[i] = __expf(row[i] - maximum) * scale;
    }
}

void Launch(float* data, std::size_t columns, unsigned rows, cudaStream_t stream) {
    if (columns <= 128) {
        ShortRows<<<(rows + 3u) / 4u, 128, 0, stream>>>(data, columns, rows);
    } else if (columns % 4 != 0 || columns > 16384) {
        LongRows<<<rows, threads, 0, stream>>>(data, columns);
    } else if (columns <= 1024) {
        CachedRows<1><<<rows, threads, 0, stream>>>(data, columns);
    } else if (columns <= 2048) {
        CachedRows<2><<<rows, threads, 0, stream>>>(data, columns);
    } else if (columns <= 4096) {
        CachedRows<4><<<rows, threads, 0, stream>>>(data, columns);
    } else if (columns <= 8192) {
        CachedRows<8><<<rows, threads, 0, stream>>>(data, columns);
    } else {
        CachedRows<16><<<rows, threads, 0, stream>>>(data, columns);
    }
    Check(cudaGetLastError(), "Launch softmax kernel");
}

struct Storage {
    int ordinal;
    cudaStream_t stream = nullptr;
    float* device = nullptr;
    std::size_t capacity = 0;

    explicit Storage(int owner) : ordinal(owner) {
        Check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "Create stream");
    }

    ~Storage() {
        int previous = ordinal;
        cudaGetDevice(&previous);
        cudaSetDevice(ordinal);
        cudaStreamSynchronize(stream);
        if (device) cudaFree(device);
        cudaStreamDestroy(stream);
        if (previous != ordinal) cudaSetDevice(previous);
    }

    void Reserve(std::size_t count) {
        if (count <= capacity) return;
        float* replacement = nullptr;
        Check(cudaMalloc(reinterpret_cast<void**>(&replacement), count * sizeof(float)), "Allocate device buffer");
        float* previous = device;
        device = replacement;
        capacity = count;
        if (previous) Check(cudaFree(previous), "Release old device buffer");
    }
};

Storage& GetStorage() {
    int ordinal = 0;
    Check(cudaGetDevice(&ordinal), "Get CUDA device");
    static thread_local std::map<int, std::unique_ptr<Storage>> devices;
    auto& storage = devices[ordinal];
    if (!storage) storage = std::make_unique<Storage>(ordinal);
    return *storage;
}

}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (row_count < 0) throw std::invalid_argument("Negative row count");
    if (row_count == 0) {
        if (input.empty()) return {};
        throw std::invalid_argument("Nonempty input requires a positive row count");
    }
    const std::size_t rows = static_cast<std::size_t>(row_count);
    if (input.size() % rows != 0) throw std::invalid_argument("Input size must be divisible by row count");
    if (input.empty()) return {};
    const std::size_t count = input.size();
    const std::size_t columns = count / rows;
    const std::size_t bytes = count * sizeof(float);
    Storage& storage = GetStorage();
    storage.Reserve(count);
    std::vector<float> result;
    std::future<std::vector<float>> allocation;

    try {
        if (count >= (1u << 20)) {
            allocation = std::async(std::launch::async, [count] {
                return std::vector<float>(count);
            });
        }
        Check(cudaMemcpyAsync(storage.device, input.data(), bytes, cudaMemcpyHostToDevice, storage.stream),
              "Upload matrix");
        Launch(storage.device, columns, static_cast<unsigned>(rows), storage.stream);
        if (allocation.valid()) {
            result = allocation.get();
        } else {
            result.resize(count);
        }
        Check(cudaMemcpyAsync(result.data(), storage.device, bytes, cudaMemcpyDeviceToHost, storage.stream),
              "Download matrix");
        Check(cudaStreamSynchronize(storage.stream), "Wait for softmax matrix");
        return result;
    } catch (...) {
        cudaStreamSynchronize(storage.stream);
        throw;
    }
}
