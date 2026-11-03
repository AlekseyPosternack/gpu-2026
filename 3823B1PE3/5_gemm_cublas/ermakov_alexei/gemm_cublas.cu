#include "gemm_cublas.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

void CheckCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

void CheckBlas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": cuBLAS status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

struct Lane {
    cudaStream_t stream = nullptr;
    cublasHandle_t handle = nullptr;

    Lane() {
        try {
            CheckCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "Create compute stream");
            CheckBlas(cublasCreate(&handle), "Create cuBLAS handle");
            CheckBlas(cublasSetStream(handle, stream), "Set cuBLAS stream");
            CheckBlas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST), "Set scalar pointer mode");
            CheckBlas(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH), "Set FP32 math mode");
        } catch (...) {
            Release();
            throw;
        }
    }

    Lane(const Lane&) = delete;
    Lane& operator=(const Lane&) = delete;

    void Release() noexcept {
        if (stream) cudaStreamSynchronize(stream);
        if (handle) cublasDestroy(handle);
        if (stream) cudaStreamDestroy(stream);
        handle = nullptr;
        stream = nullptr;
    }

    ~Lane() { Release(); }

    void Multiply(const float* a, const float* b, float* c, int n, int rows) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        CheckBlas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              n, rows, n, &alpha, b, n, a, n, &beta, c, n),
                  "Multiply matrix strip");
    }
};

struct Storage {
    int ordinal;
    std::array<Lane, 4> lanes;
    cudaStream_t upload = nullptr;
    cudaEvent_t ready = nullptr;
    float* data = nullptr;
    std::size_t capacity = 0;

    explicit Storage(int device) : ordinal(device) {
        try {
            CheckCuda(cudaStreamCreateWithFlags(&upload, cudaStreamNonBlocking), "Create upload stream");
            CheckCuda(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming), "Create upload event");
        } catch (...) {
            if (upload) cudaStreamDestroy(upload);
            throw;
        }
    }

    void Wait() noexcept {
        cudaStreamSynchronize(upload);
        for (auto& lane : lanes) cudaStreamSynchronize(lane.stream);
    }

    ~Storage() {
        int previous = ordinal;
        cudaGetDevice(&previous);
        cudaSetDevice(ordinal);
        Wait();
        for (auto& lane : lanes) lane.Release();
        if (data) cudaFree(data);
        cudaEventDestroy(ready);
        cudaStreamDestroy(upload);
        if (previous != ordinal) cudaSetDevice(previous);
    }

    void Reserve(std::size_t count) {
        if (count <= capacity) return;
        float* replacement = nullptr;
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&replacement), 3 * count * sizeof(float)),
                  "Allocate matrices");
        float* previous = data;
        data = replacement;
        capacity = count;
        if (previous) CheckCuda(cudaFree(previous), "Release old matrices");
    }
};

Storage& GetStorage() {
    int ordinal = 0;
    CheckCuda(cudaGetDevice(&ordinal), "Get CUDA device");
    static thread_local std::map<int, std::unique_ptr<Storage>> devices;
    auto& storage = devices[ordinal];
    if (!storage) storage = std::make_unique<Storage>(ordinal);
    return *storage;
}

}

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
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
        if (n >= 1024 && n % 4 == 0) {
            auto allocation = std::async(std::launch::async, [count] {
                return std::vector<float>(count);
            });
            CheckCuda(cudaMemcpyAsync(db, b.data(), bytes, cudaMemcpyHostToDevice, storage.upload), "Upload B");
            CheckCuda(cudaEventRecord(storage.ready, storage.upload), "Record B upload");
            const int rows = n / 4;
            const std::size_t strip = static_cast<std::size_t>(rows) * n;
            for (int part = 0; part < 4; ++part) {
                const std::size_t offset = part * strip;
                Lane& lane = storage.lanes[part];
                CheckCuda(cudaMemcpyAsync(da + offset, a.data() + offset, strip * sizeof(float),
                                          cudaMemcpyHostToDevice, lane.stream), "Upload A strip");
                CheckCuda(cudaStreamWaitEvent(lane.stream, storage.ready, 0), "Wait for B upload");
                lane.Multiply(da + offset, db, dc + offset, n, rows);
            }
            result = allocation.get();
            for (int part = 0; part < 4; ++part) {
                const std::size_t offset = part * strip;
                CheckCuda(cudaMemcpyAsync(result.data() + offset, dc + offset, strip * sizeof(float),
                                          cudaMemcpyDeviceToHost, storage.lanes[part].stream), "Download C strip");
            }
            for (auto& lane : storage.lanes) {
                CheckCuda(cudaStreamSynchronize(lane.stream), "Wait for cuBLAS strip");
            }
            return result;
        }

        Lane& lane = storage.lanes[0];
        CheckCuda(cudaMemcpyAsync(da, a.data(), bytes, cudaMemcpyHostToDevice, lane.stream), "Upload A");
        CheckCuda(cudaMemcpyAsync(db, b.data(), bytes, cudaMemcpyHostToDevice, lane.stream), "Upload B");
        lane.Multiply(da, db, dc, n, n);
        result.resize(count);
        CheckCuda(cudaMemcpyAsync(result.data(), dc, bytes, cudaMemcpyDeviceToHost, lane.stream), "Download C");
        CheckCuda(cudaStreamSynchronize(lane.stream), "Wait for cuBLAS");
        return result;
    } catch (...) {
        storage.Wait();
        throw;
    }
}
