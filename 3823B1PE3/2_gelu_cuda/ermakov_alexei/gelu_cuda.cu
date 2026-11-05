#include "gelu_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <future>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

constexpr std::size_t tile_size = 1u << 21;
constexpr int lane_count = 3;
constexpr unsigned threads = 256;

void Check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

__device__ __forceinline__ float Activate(float x) {
    const float exponent = 1.5957691216057308f * x *
                           fmaf(0.044715f * x, x, 1.0f);
    const float e = __expf(-fabsf(exponent));
    const float inverse = __fdividef(1.0f, 1.0f + e);
    return x >= 0.0f ? x * inverse : (x * e) * inverse;
}

__global__ void Transform(float* values, std::size_t count) {
    const std::size_t first = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                              threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    float2* pairs = reinterpret_cast<float2*>(values);
    for (std::size_t i = first; i < count / 2; i += stride) {
        float2 pair = pairs[i];
        pair.x = Activate(pair.x);
        pair.y = Activate(pair.y);
        pairs[i] = pair;
    }
    if (first == 0 && count % 2 != 0) {
        values[count - 1] = Activate(values[count - 1]);
    }
}

struct Lane {
    cudaStream_t stream = nullptr;
    float* host = nullptr;
    float* device = nullptr;
    std::size_t capacity = 0;

    Lane() = default;
    Lane(const Lane&) = delete;
    Lane& operator=(const Lane&) = delete;

    void Release() noexcept {
        if (stream) cudaStreamSynchronize(stream);
        if (device) cudaFree(device);
        if (host) cudaFreeHost(host);
        if (stream) cudaStreamDestroy(stream);
        stream = nullptr;
        host = nullptr;
        device = nullptr;
        capacity = 0;
    }

    ~Lane() { Release(); }

    void Reserve(std::size_t count) {
        if (count <= capacity) return;
        Release();
        Check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              "Create stream");
        Check(cudaMallocHost(reinterpret_cast<void**>(&host), count * sizeof(float)),
              "Allocate pinned staging memory");
        Check(cudaMalloc(reinterpret_cast<void**>(&device), count * sizeof(float)),
              "Allocate device staging memory");
        capacity = count;
    }

    void Submit(const float* source, float* destination, std::size_t count) {
        const std::size_t bytes = count * sizeof(float);
        std::memcpy(host, source, bytes);
        Check(cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice, stream),
              "Upload tile");
        const std::size_t pairs = (count + 1) / 2;
        const unsigned blocks = static_cast<unsigned>(
            std::min<std::size_t>((pairs + threads - 1) / threads, 4096));
        Transform<<<blocks, threads, 0, stream>>>(device, count);
        Check(cudaGetLastError(), "Launch GELU kernel");
        Check(cudaMemcpyAsync(destination, device, bytes, cudaMemcpyDeviceToHost, stream),
              "Download tile");
    }
};

struct Pipeline {
    int ordinal;
    std::array<Lane, lane_count> lanes;
    float* downloaded = nullptr;
    std::size_t capacity = 0;

    explicit Pipeline(int device) : ordinal(device) {}

    ~Pipeline() {
        int previous = ordinal;
        cudaGetDevice(&previous);
        cudaSetDevice(ordinal);
        for (auto& lane : lanes) lane.Release();
        if (downloaded) cudaFreeHost(downloaded);
        if (previous != ordinal) cudaSetDevice(previous);
    }

    void Wait() noexcept {
        for (auto& lane : lanes) {
            if (lane.stream) cudaStreamSynchronize(lane.stream);
        }
    }

    void Reserve(std::size_t count) {
        if (count <= capacity) return;
        float* replacement = nullptr;
        Check(cudaMallocHost(reinterpret_cast<void**>(&replacement), count * sizeof(float)),
              "Allocate download buffer");
        if (downloaded) cudaFreeHost(downloaded);
        downloaded = replacement;
        capacity = count;
    }
};

Pipeline& GetPipeline() {
    int ordinal = 0;
    Check(cudaGetDevice(&ordinal), "Get CUDA device");
    static thread_local std::map<int, std::unique_ptr<Pipeline>> pipelines;
    auto& pipeline = pipelines[ordinal];
    if (!pipeline) pipeline = std::make_unique<Pipeline>(ordinal);
    return *pipeline;
}

}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    if (input.empty()) return {};

    const std::size_t count = input.size();
    const std::size_t tiles = 1 + (count - 1) / tile_size;
    const std::size_t active = std::min<std::size_t>(tiles, lane_count);
    Pipeline& pipeline = GetPipeline();

    try {
        if (tiles == 1) {
            Lane& lane = pipeline.lanes[0];
            lane.Reserve(count);
            lane.Submit(input.data(), lane.host, count);
            std::vector<float> result(count);
            Check(cudaStreamSynchronize(lane.stream), "Wait for result");
            std::memcpy(result.data(), lane.host, count * sizeof(float));
            return result;
        }

        pipeline.Reserve(count);
        auto allocation = std::async(std::launch::async, [count] {
            return std::vector<float>(count);
        });
        for (std::size_t i = 0; i < active; ++i) {
            const std::size_t offset = i * tile_size;
            const std::size_t length = std::min(tile_size, count - offset);
            pipeline.lanes[i].Reserve(length);
            pipeline.lanes[i].Submit(input.data() + offset, pipeline.downloaded + offset, length);
        }

        for (std::size_t tile = 0; tile < tiles; ++tile) {
            Lane& lane = pipeline.lanes[tile % active];
            Check(cudaStreamSynchronize(lane.stream), "Wait for tile");

            const std::size_t next = tile + active;
            if (next < tiles) {
                const std::size_t next_offset = next * tile_size;
                const std::size_t next_length = std::min(tile_size, count - next_offset);
                lane.Submit(input.data() + next_offset, pipeline.downloaded + next_offset,
                            next_length);
            }
        }
        auto result = allocation.get();
        std::memcpy(result.data(), pipeline.downloaded, count * sizeof(float));
        return result;
    } catch (...) {
        pipeline.Wait();
        throw;
    }
}
