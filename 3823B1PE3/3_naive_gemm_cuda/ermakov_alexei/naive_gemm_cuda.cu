#include "naive_gemm_cuda.h"

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

__device__ __forceinline__ void Accumulate(float4& result, float a, float4 b) {
    result.x = fmaf(a, b.x, result.x);
    result.y = fmaf(a, b.y, result.y);
    result.z = fmaf(a, b.z, result.z);
    result.w = fmaf(a, b.w, result.w);
}

template<int Rows, int Groups, int RowGroup = 8>
__global__ void __launch_bounds__(128) MultiplyVectors(
    const float* __restrict__ a, const float4* __restrict__ b,
    float4* __restrict__ c, int n, int rows) {
    const int row_tiles = rows / (4 * Rows);
    const int column_tiles = n / (128 * Groups);
    const int group = blockIdx.x / (RowGroup * column_tiles);
    const int row_start = group * RowGroup;
    const int group_rows = min(RowGroup, row_tiles - row_start);
    const int local = blockIdx.x - group * RowGroup * column_tiles;
    const int row = (row_start + local % group_rows) * (4 * Rows) + threadIdx.y;
    const int column = (local / group_rows) * (32 * Groups) + threadIdx.x;
    const std::size_t pitch = static_cast<std::size_t>(n) / 4;
    float4 sums[Rows][Groups] = {};

    for (int k = 0; k < n; k += 4) {
        float4 bv[Groups][4];
#pragma unroll
        for (int g = 0; g < Groups; ++g) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                bv[g][j] = b[static_cast<std::size_t>(k + j) * pitch + column + g * 32];
            }
        }
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            const float4 av = *reinterpret_cast<const float4*>(
                a + static_cast<std::size_t>(row + 4 * r) * n + k);
#pragma unroll
            for (int g = 0; g < Groups; ++g) {
                Accumulate(sums[r][g], av.x, bv[g][0]);
                Accumulate(sums[r][g], av.y, bv[g][1]);
                Accumulate(sums[r][g], av.z, bv[g][2]);
                Accumulate(sums[r][g], av.w, bv[g][3]);
            }
        }
    }

#pragma unroll
    for (int r = 0; r < Rows; ++r) {
#pragma unroll
        for (int g = 0; g < Groups; ++g) {
            c[static_cast<std::size_t>(row + 4 * r) * pitch + column + g * 32] = sums[r][g];
        }
    }
}

__global__ void MultiplySmall(const float* __restrict__ a,
                              const float* __restrict__ b,
                              float* __restrict__ c, int n) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n || column >= n) return;

    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
        sum = fmaf(a[static_cast<std::size_t>(row) * n + k],
                   b[static_cast<std::size_t>(k) * n + column], sum);
    }
    c[static_cast<std::size_t>(row) * n + column] = sum;
}

struct Storage {
    int ordinal;
    cudaStream_t stream = nullptr;
    std::array<cudaStream_t, 4> workers{};
    cudaEvent_t ready = nullptr;
    float* data = nullptr;
    std::size_t capacity = 0;

    explicit Storage(int device) : ordinal(device) {
        try {
            Check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "Create upload stream");
            for (auto& worker : workers) {
                Check(cudaStreamCreateWithFlags(&worker, cudaStreamNonBlocking), "Create compute stream");
            }
            Check(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming), "Create upload event");
        } catch (...) {
            for (auto worker : workers) if (worker) cudaStreamDestroy(worker);
            if (stream) cudaStreamDestroy(stream);
            throw;
        }
    }

    ~Storage() {
        int previous = ordinal;
        cudaGetDevice(&previous);
        cudaSetDevice(ordinal);
        Wait();
        if (data) cudaFree(data);
        cudaEventDestroy(ready);
        for (auto worker : workers) cudaStreamDestroy(worker);
        cudaStreamDestroy(stream);
        if (previous != ordinal) cudaSetDevice(previous);
    }

    void Wait() noexcept {
        cudaStreamSynchronize(stream);
        for (auto worker : workers) cudaStreamSynchronize(worker);
    }

    void Reserve(std::size_t count) {
        if (count <= capacity) return;
        float* replacement = nullptr;
        Check(cudaMalloc(reinterpret_cast<void**>(&replacement), 3 * count * sizeof(float)),
              "Allocate matrices");
        if (data) Check(cudaFree(data), "Release old matrices");
        data = replacement;
        capacity = count;
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

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
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
        if (n >= 1024 && n % 256 == 0) {
            Check(cudaMemcpyAsync(db, b.data(), bytes, cudaMemcpyHostToDevice, storage.stream),
                  "Upload B");
            Check(cudaEventRecord(storage.ready, storage.stream), "Record B upload");
            const int rows = n / 4;
            const std::size_t strip = static_cast<std::size_t>(rows) * n;
            const unsigned blocks = static_cast<unsigned>((rows / 64) * (dimension / 256));
            for (int part = 0; part < 4; ++part) {
                const std::size_t offset = part * strip;
                const auto worker = storage.workers[part];
                Check(cudaMemcpyAsync(da + offset, a.data() + offset, strip * sizeof(float),
                                      cudaMemcpyHostToDevice, worker), "Upload A strip");
                Check(cudaStreamWaitEvent(worker, storage.ready, 0), "Wait for B upload");
                MultiplyVectors<16, 2><<<blocks, dim3(32, 4), 0, worker>>>(
                    da + offset, reinterpret_cast<const float4*>(db),
                    reinterpret_cast<float4*>(dc + offset), n, rows);
                Check(cudaGetLastError(), "Launch matrix strip multiplication");
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

        Check(cudaMemcpyAsync(da, a.data(), bytes, cudaMemcpyHostToDevice, storage.stream),
              "Upload A");
        Check(cudaMemcpyAsync(db, b.data(), bytes, cudaMemcpyHostToDevice, storage.stream),
              "Upload B");
        if (n % 128 == 0) {
            const dim3 block(32, 4);
            const unsigned blocks = static_cast<unsigned>((dimension / 8) * (dimension / 128));
            MultiplyVectors<2, 1><<<blocks, block, 0, storage.stream>>>(
                da, reinterpret_cast<const float4*>(db), reinterpret_cast<float4*>(dc), n, n);
        } else {
            const dim3 block(32, 4);
            const dim3 grid((dimension + 31) / 32, (dimension + 3) / 4);
            MultiplySmall<<<grid, block, 0, storage.stream>>>(da, db, dc, n);
        }
        Check(cudaGetLastError(), "Launch matrix multiplication");

        result.resize(count);
        Check(cudaMemcpyAsync(result.data(), dc, bytes, cudaMemcpyDeviceToHost, storage.stream),
              "Download C");
        Check(cudaStreamSynchronize(storage.stream), "Wait for matrix multiplication");
        return result;
    } catch (...) {
        storage.Wait();
        throw;
    }
}
