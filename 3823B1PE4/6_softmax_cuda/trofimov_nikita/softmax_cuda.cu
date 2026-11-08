#include "softmax_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace {

constexpr int BLOCK_SIZE = 256;

__device__ __forceinline__ float WarpReduceMax(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(
            value,
            __shfl_xor_sync(0xffffffff, value, offset)
        );
    }

    return value;
}

__device__ __forceinline__ float WarpReduceSum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffff, value, offset);
    }

    return value;
}

__device__ __forceinline__ float BlockReduceMax(
    float value,
    float* shared
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    value = WarpReduceMax(value);

    if (lane == 0) {
        shared[warp] = value;
    }

    __syncthreads();

    if (warp == 0) {
        value = (lane < (blockDim.x >> 5))
            ? shared[lane]
            : -INFINITY;

        value = WarpReduceMax(value);

        if (lane == 0) {
            shared[0] = value;
        }
    }

    __syncthreads();

    return shared[0];
}

__device__ __forceinline__ float BlockReduceSum(
    float value,
    float* shared
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    value = WarpReduceSum(value);

    if (lane == 0) {
        shared[warp] = value;
    }

    __syncthreads();

    if (warp == 0) {
        value = (lane < (blockDim.x >> 5))
            ? shared[lane]
            : 0.0f;

        value = WarpReduceSum(value);

        if (lane == 0) {
            shared[0] = value;
        }
    }

    __syncthreads();

    return shared[0];
}

__global__ void SoftmaxVectorKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int row_size
) {
    __shared__ float shared[8];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const float4* src = reinterpret_cast<const float4*>(
        input + static_cast<std::size_t>(row) * row_size
    );

    float4* dst = reinterpret_cast<float4*>(
        output + static_cast<std::size_t>(row) * row_size
    );

    const int count = row_size / 4;

    float local_max = -INFINITY;

    for (int i = tid; i < count; i += blockDim.x) {
        const float4 v = src[i];

        local_max = fmaxf(
            local_max,
            fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w))
        );
    }

    const float row_max = BlockReduceMax(local_max, shared);

    float local_sum = 0.0f;

    for (int i = tid; i < count; i += blockDim.x) {
        const float4 v = src[i];

        float4 e;

        e.x = __expf(v.x - row_max);
        e.y = __expf(v.y - row_max);
        e.z = __expf(v.z - row_max);
        e.w = __expf(v.w - row_max);

        dst[i] = e;

        local_sum += e.x + e.y + e.z + e.w;
    }

    const float row_sum = BlockReduceSum(local_sum, shared);

    const float inv_sum = 1.0f / row_sum;

    for (int i = tid; i < count; i += blockDim.x) {
        float4 e = dst[i];

        e.x *= inv_sum;
        e.y *= inv_sum;
        e.z *= inv_sum;
        e.w *= inv_sum;

        dst[i] = e;
    }
}

__global__ void SoftmaxScalarKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int row_size
) {
    __shared__ float shared[8];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const float* src = input +
        static_cast<std::size_t>(row) * row_size;

    float* dst = output +
        static_cast<std::size_t>(row) * row_size;

    float local_max = -INFINITY;

    for (int i = tid; i < row_size; i += blockDim.x) {
        local_max = fmaxf(local_max, src[i]);
    }

    const float row_max = BlockReduceMax(local_max, shared);

    float local_sum = 0.0f;

    for (int i = tid; i < row_size; i += blockDim.x) {
        const float e = __expf(src[i] - row_max);

        dst[i] = e;
        local_sum += e;
    }

    const float row_sum = BlockReduceSum(local_sum, shared);

    const float inv_sum = 1.0f / row_sum;

    for (int i = tid; i < row_size; i += blockDim.x) {
        dst[i] *= inv_sum;
    }
}

void CheckCUDA(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(error)
        );
    }
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) {
        CheckCUDA(
            cudaMalloc(reinterpret_cast<void**>(&ptr), bytes),
            "cudaMalloc"
        );
    }

    ~DeviceBuffer() {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    float* get() const {
        return ptr;
    }

private:
    float* ptr = nullptr;
};

} // namespace

std::vector<float> SoftmaxCUDA(
    const std::vector<float>& input,
    int row_count
) {
    if (row_count <= 0 || input.empty()) {
        return {};
    }

    if (input.size() % static_cast<std::size_t>(row_count) != 0) {
        throw std::invalid_argument(
            "Input size must be divisible by row_count"
        );
    }

    const std::size_t row_size =
        input.size() / static_cast<std::size_t>(row_count);

    if (row_size == 0 ||
        row_size > static_cast<std::size_t>(INT_MAX)) {
        throw std::invalid_argument("Invalid row size");
    }

    const int cols = static_cast<int>(row_size);

    const std::size_t count = input.size();
    const std::size_t bytes = count * sizeof(float);

    std::vector<float> output(count);

    DeviceBuffer deviceMemory(2 * bytes);

    float* d_input = deviceMemory.get();
    float* d_output = d_input + count;

    CheckCUDA(
        cudaMemcpy(
            d_input,
            input.data(),
            bytes,
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy H2D"
    );

    if (cols % 4 == 0) {
        SoftmaxVectorKernel<<<row_count, BLOCK_SIZE>>>(
            d_input,
            d_output,
            cols
        );
    } else {
        SoftmaxScalarKernel<<<row_count, BLOCK_SIZE>>>(
            d_input,
            d_output,
            cols
        );
    }

    CheckCUDA(cudaGetLastError(), "Softmax kernel launch");

    CheckCUDA(
        cudaMemcpy(
            output.data(),
            d_output,
            bytes,
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy D2H"
    );

    return output;
}