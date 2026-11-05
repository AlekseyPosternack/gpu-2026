#include "block_gemm_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

constexpr int SmallTileSize = 16;
constexpr int LargeTileSize = 32;

void CheckCudaError(const cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(error)
        );
    }
}

struct DeviceMemoryDeleter {
    void operator()(float* pointer) const noexcept {
        cudaFree(pointer);
    }
};

template <int TileSize, bool CheckBounds>
__global__ void BlockGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                const std::size_t n) {
    __shared__ float tileA[TileSize][TileSize];
    __shared__ float tileB[TileSize][TileSize];
    __shared__ float tileC[TileSize][TileSize];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const std::size_t row =
        static_cast<std::size_t>(blockIdx.y) * TileSize + ty;
    const std::size_t column =
        static_cast<std::size_t>(blockIdx.x) * TileSize + tx;

    // Каждый поток владеет одним элементом блока C.
    tileC[ty][tx] = 0.0f;

    for (std::size_t offset = 0; offset < n; offset += TileSize) {
        // Соседние потоки читают соседние элементы строк A и B.
        if constexpr (CheckBounds) {
            tileA[ty][tx] = (row < n && offset + tx < n)
                ? a[row * n + offset + tx] : 0.0f;
            tileB[ty][tx] = (offset + ty < n && column < n)
                ? b[(offset + ty) * n + column] : 0.0f;
        } else {
            tileA[ty][tx] = a[row * n + offset + tx];
            tileB[ty][tx] = b[(offset + ty) * n + column];
        }

        // Все потоки участвуют в обоих барьерах, включая граничные.
        __syncthreads();

        // Между блоками K результат хранится в shared memory;
        // внутри блока K сумма накапливается в регистре.
        float sum = tileC[ty][tx];
        #pragma unroll
        for (int k = 0; k < TileSize; ++k) {
            sum = fmaf(tileA[ty][k], tileB[k][tx], sum);
        }
        tileC[ty][tx] = sum;

        // Следующий блок A/B можно загружать только после всех чтений.
        __syncthreads();
    }

    if constexpr (CheckBounds) {
        if (row < n && column < n) {
            c[row * n + column] = tileC[ty][tx];
        }
    } else {
        c[row * n + column] = tileC[ty][tx];
    }
}

template <int TileSize>
void LaunchBlockGemm(const float* a, const float* b, float* c,
                     const std::size_t n) {
    const dim3 block(TileSize, TileSize);
    const unsigned int blockCount = static_cast<unsigned int>(
        (n + TileSize - 1) / TileSize
    );
    const dim3 grid(blockCount, blockCount);

    // Для полных блоков проверки границ удаляются при компиляции.
    if (n % TileSize == 0) {
        BlockGemmKernel<TileSize, false><<<grid, block>>>(a, b, c, n);
    } else {
        BlockGemmKernel<TileSize, true><<<grid, block>>>(a, b, c, n);
    }
}

} // namespace

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                               const std::vector<float>& b,
                               const int n) {
    if (n < 0) {
        throw std::invalid_argument("Matrix size must be non-negative");
    }

    const std::size_t size = static_cast<std::size_t>(n);
    const std::size_t maxSize = std::numeric_limits<std::size_t>::max();
    if (size != 0 && size > maxSize / size) {
        throw std::length_error("Matrix size is too large");
    }

    const std::size_t elementCount = size * size;
    if (a.size() != elementCount || b.size() != elementCount) {
        throw std::invalid_argument("Both matrices must contain n*n elements");
    }
    if (elementCount == 0) {
        return {};
    }
    if (elementCount > maxSize / sizeof(float) / 3) {
        throw std::length_error("Device allocation size is too large");
    }

    const std::size_t bytes = elementCount * sizeof(float);
    void* allocation = nullptr;
    // Один буфер содержит A, B и C и освобождается также при исключениях.
    CheckCudaError(cudaMalloc(&allocation, 3 * bytes),
                   "Failed to allocate device matrices");
    const std::unique_ptr<float, DeviceMemoryDeleter> deviceMemory(
        static_cast<float*>(allocation)
    );
    float* const deviceA = deviceMemory.get();
    float* const deviceB = deviceA + elementCount;
    float* const deviceC = deviceB + elementCount;

    CheckCudaError(cudaMemcpy(deviceA, a.data(), bytes, cudaMemcpyHostToDevice),
                   "Failed to copy A to GPU");
    CheckCudaError(cudaMemcpy(deviceB, b.data(), bytes, cudaMemcpyHostToDevice),
                   "Failed to copy B to GPU");

    if (n < LargeTileSize) {
        LaunchBlockGemm<SmallTileSize>(deviceA, deviceB, deviceC, size);
    } else {
        LaunchBlockGemm<LargeTileSize>(deviceA, deviceB, deviceC, size);
    }
    CheckCudaError(cudaGetLastError(), "Failed to launch block GEMM kernel");

    // Выделение памяти результата CPU может перекрываться с работой ядра.
    std::vector<float> result(elementCount);
    // Копирование в том же default stream ожидает завершения вычислений.
    CheckCudaError(cudaMemcpy(result.data(), deviceC, bytes, cudaMemcpyDeviceToHost),
                   "Failed to copy C from GPU");

    return result;
}
