#include "block_gemm_cuda.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

void Check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

template<int Height, int Depth>
__device__ __forceinline__ void ReadPanels(
    const float* a, const float* b, int n, int row, int column, int start,
    int thread, float4* av, float4* bv) {
#pragma unroll
    for (int i = 0; i < (Height * Depth / 4 + 255) / 256; ++i) {
        const int index = thread + i * 256;
        if (index < Height * Depth / 4) {
            const int r = index / (Depth / 4);
            const int k = (index % (Depth / 4)) * 4;
            av[i] = *reinterpret_cast<const float4*>(
                a + static_cast<std::size_t>(row + r) * n + start + k);
        }
    }
#pragma unroll
    for (int i = 0; i < (Depth * 32 + 255) / 256; ++i) {
        const int index = thread + i * 256;
        if (index < Depth * 32) {
            const int k = index / 32;
            const int c = (index % 32) * 4;
            bv[i] = *reinterpret_cast<const float4*>(
                b + static_cast<std::size_t>(start + k) * n + column + c);
        }
    }
}

template<int Height, int Depth>
__device__ __forceinline__ void StorePanels(
    float* a, float* b, int thread, const float4* av, const float4* bv) {
#pragma unroll
    for (int i = 0; i < (Height * Depth / 4 + 255) / 256; ++i) {
        const int index = thread + i * 256;
        if (index < Height * Depth / 4) {
            const int r = index / (Depth / 4);
            const int k = (index % (Depth / 4)) * 4;
            const int swizzled_row = r ^ ((k / 4) * (128 / Depth));
            a[(k + 0) * Height + swizzled_row] = av[i].x;
            a[(k + 1) * Height + swizzled_row] = av[i].y;
            a[(k + 2) * Height + swizzled_row] = av[i].z;
            a[(k + 3) * Height + swizzled_row] = av[i].w;
        }
    }
#pragma unroll
    for (int i = 0; i < (Depth * 32 + 255) / 256; ++i) {
        const int index = thread + i * 256;
        if (index < Depth * 32) {
            reinterpret_cast<float4*>(b)[index] = bv[i];
        }
    }
}

__device__ __forceinline__ void Accumulate(float4& result, float a, float4 b) {
    result.x = fmaf(a, b.x, result.x);
    result.y = fmaf(a, b.y, result.y);
    result.z = fmaf(a, b.z, result.z);
    result.w = fmaf(a, b.w, result.w);
}

template<int Height, int Depth>
__global__ void __launch_bounds__(256) MultiplyPanels(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ c, int n) {
    __shared__ __align__(16) float left[2][Depth][Height];
    __shared__ __align__(16) float right[2][Depth][128];
    constexpr int rows_per_thread = Height / 16;
    const int thread = threadIdx.y * 16 + threadIdx.x;
    const int row = blockIdx.y * Height;
    const int column = blockIdx.x * 128;
    const int local_row = threadIdx.y * rows_per_thread;
    const int local_column = threadIdx.x * 4;
    float4 sums[rows_per_thread][2] = {};
    float4 av[(Height * Depth / 4 + 255) / 256];
    float4 bv[(Depth * 32 + 255) / 256];

    ReadPanels<Height, Depth>(a, b, n, row, column, 0, thread, av, bv);
    StorePanels<Height, Depth>(&left[0][0][0], &right[0][0][0], thread, av, bv);
    __syncthreads();

    int buffer = 0;
    for (int start = 0; start < n; start += Depth) {
        const bool more = start + Depth < n;
        if (more) {
            ReadPanels<Height, Depth>(a, b, n, row, column, start + Depth,
                                      thread, av, bv);
        }
#pragma unroll
        for (int k = 0; k < Depth; ++k) {
            const float4 b0 = *reinterpret_cast<const float4*>(&right[buffer][k][local_column]);
            const float4 b1 = *reinterpret_cast<const float4*>(&right[buffer][k][local_column + 64]);
#pragma unroll
            for (int r = 0; r < rows_per_thread; r += 4) {
                const int swizzled_row = (local_row + r) ^ ((k / 4) * (128 / Depth));
                const float4 values = *reinterpret_cast<const float4*>(&left[buffer][k][swizzled_row]);
                const float scalars[4] = {values.x, values.y, values.z, values.w};
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    Accumulate(sums[r + i][0], scalars[i], b0);
                    Accumulate(sums[r + i][1], scalars[i], b1);
                }
            }
        }
        if (more) {
            buffer ^= 1;
            StorePanels<Height, Depth>(&left[buffer][0][0], &right[buffer][0][0], thread, av, bv);
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < rows_per_thread; ++r) {
        float* output = c + static_cast<std::size_t>(row + local_row + r) * n +
                        column + local_column;
        *reinterpret_cast<float4*>(output) = sums[r][0];
        *reinterpret_cast<float4*>(output + 64) = sums[r][1];
    }
}

__global__ void MultiplyEdges(const float* a, const float* b, float* c, int n) {
    __shared__ float left[16][16];
    __shared__ float right[16][16];
    const int row = blockIdx.y * 16 + threadIdx.y;
    const int column = blockIdx.x * 16 + threadIdx.x;
    float sum = 0.0f;
    for (int start = 0; start < n; start += 16) {
        left[threadIdx.y][threadIdx.x] = 0.0f;
        right[threadIdx.y][threadIdx.x] = 0.0f;
        if (row < n && start + threadIdx.x < n) {
            left[threadIdx.y][threadIdx.x] = a[static_cast<std::size_t>(row) * n + start + threadIdx.x];
        }
        if (start + threadIdx.y < n && column < n) {
            right[threadIdx.y][threadIdx.x] = b[static_cast<std::size_t>(start + threadIdx.y) * n + column];
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            sum = fmaf(left[threadIdx.y][k], right[k][threadIdx.x], sum);
        }
        __syncthreads();
    }
    if (row < n && column < n) {
        c[static_cast<std::size_t>(row) * n + column] = sum;
    }
}

struct Storage {
    int ordinal;
    cudaStream_t upload = nullptr;
    std::array<cudaStream_t, 4> workers{};
    cudaEvent_t ready = nullptr;
    float* data = nullptr;
    std::size_t capacity = 0;

    explicit Storage(int device) : ordinal(device) {
        try {
            Check(cudaStreamCreateWithFlags(&upload, cudaStreamNonBlocking), "Create upload stream");
            for (auto& worker : workers) {
                Check(cudaStreamCreateWithFlags(&worker, cudaStreamNonBlocking), "Create compute stream");
            }
            Check(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming), "Create upload event");
        } catch (...) {
            for (auto worker : workers) if (worker) cudaStreamDestroy(worker);
            if (upload) cudaStreamDestroy(upload);
            throw;
        }
    }

    void Wait() noexcept {
        cudaStreamSynchronize(upload);
        for (auto worker : workers) cudaStreamSynchronize(worker);
    }

    ~Storage() {
        int previous = ordinal;
        cudaGetDevice(&previous);
        cudaSetDevice(ordinal);
        Wait();
        if (data) cudaFree(data);
        cudaEventDestroy(ready);
        for (auto worker : workers) cudaStreamDestroy(worker);
        cudaStreamDestroy(upload);
        if (previous != ordinal) cudaSetDevice(previous);
    }

    void Reserve(std::size_t count) {
        if (count <= capacity) return;
        float* replacement = nullptr;
        Check(cudaMalloc(reinterpret_cast<void**>(&replacement), 3 * count * sizeof(float)),
              "Allocate matrices");
        float* previous = data;
        data = replacement;
        capacity = count;
        if (previous) Check(cudaFree(previous), "Release old matrices");
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

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                               const std::vector<float>& b, int n) {
    if (n < 0) throw std::invalid_argument("Negative matrix dimension");
    const std::size_t dimension = static_cast<std::size_t>(n);
    if (dimension != 0 && dimension >
        std::numeric_limits<std::size_t>::max() / sizeof(float) / 3 / dimension) {
        throw std::length_error("Matrix dimension is too large");
    }
    const std::size_t count = dimension * dimension;
    if (a.size() != count || b.size() != count) {
        throw std::invalid_argument("Expected n*n elements in both matrices");
    }
    if (count == 0) return {};

    Storage& storage = GetStorage();
    storage.Reserve(count);
    const std::size_t bytes = count * sizeof(float);
    float* da = storage.data;
    float* db = da + count;
    float* dc = db + count;
    std::vector<float> result;

    try {
        if (n >= 1024 && n % 512 == 0) {
            Check(cudaMemcpyAsync(db, b.data(), bytes, cudaMemcpyHostToDevice, storage.upload), "Upload B");
            Check(cudaEventRecord(storage.ready, storage.upload), "Record B upload");
            const int rows = n / 4;
            const std::size_t strip = static_cast<std::size_t>(rows) * n;
            const dim3 grid(n / 128, rows / 128);
            for (int part = 0; part < 4; ++part) {
                const std::size_t offset = part * strip;
                const auto worker = storage.workers[part];
                Check(cudaMemcpyAsync(da + offset, a.data() + offset, strip * sizeof(float),
                                      cudaMemcpyHostToDevice, worker), "Upload A strip");
                Check(cudaStreamWaitEvent(worker, storage.ready, 0), "Wait for B upload");
                MultiplyPanels<128, 8><<<grid, dim3(16, 16), 0, worker>>>(da + offset, db, dc + offset, n);
                Check(cudaGetLastError(), "Launch block multiplication");
            }
            result.resize(count);
            for (int part = 0; part < 4; ++part) {
                const std::size_t offset = part * strip;
                Check(cudaMemcpyAsync(result.data() + offset, dc + offset, strip * sizeof(float),
                                      cudaMemcpyDeviceToHost, storage.workers[part]), "Download C strip");
            }
            for (auto worker : storage.workers) {
                Check(cudaStreamSynchronize(worker), "Wait for matrix strip");
            }
            return result;
        }

        Check(cudaMemcpyAsync(da, a.data(), bytes, cudaMemcpyHostToDevice, storage.upload), "Upload A");
        Check(cudaMemcpyAsync(db, b.data(), bytes, cudaMemcpyHostToDevice, storage.upload), "Upload B");
        if (n % 128 == 0) {
            const dim3 grid(n / 128, n / 64);
            MultiplyPanels<64, 8><<<grid, dim3(16, 16), 0, storage.upload>>>(da, db, dc, n);
        } else {
            const dim3 grid((dimension + 15) / 16, (dimension + 15) / 16);
            MultiplyEdges<<<grid, dim3(16, 16), 0, storage.upload>>>(da, db, dc, n);
        }
        Check(cudaGetLastError(), "Launch block multiplication");
        result.resize(count);
        Check(cudaMemcpyAsync(result.data(), dc, bytes, cudaMemcpyDeviceToHost, storage.upload), "Download C");
        Check(cudaStreamSynchronize(storage.upload), "Wait for block multiplication");
        return result;
    } catch (...) {
        storage.Wait();
        throw;
    }
}
