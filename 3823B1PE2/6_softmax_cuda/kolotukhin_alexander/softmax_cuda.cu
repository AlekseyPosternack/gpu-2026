#include "softmax_cuda.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t error_code = (call);                                 \
        if (error_code != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(error_code)); \
            std::exit(EXIT_FAILURE);                                     \
        }                                                                \
    } while (0)

namespace {

constexpr int BLOCK_THREADS = 256;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = BLOCK_THREADS / WARP_SIZE;

__device__ __forceinline__ float block_reduce_max(float thread_value, int thread_id, float* shared_memory) {
    #pragma unroll
    for (int shuffle_offset = 16; shuffle_offset > 0; shuffle_offset /= 2) {
        thread_value = fmaxf(thread_value, __shfl_down_sync(0xffffffff, thread_value, shuffle_offset));
    }

    if (thread_id % WARP_SIZE == 0) {
        shared_memory[thread_id / WARP_SIZE] = thread_value;
    }

    __syncthreads();

    if (thread_id < WARP_SIZE) {
        thread_value = (thread_id < WARPS_PER_BLOCK) ? shared_memory[thread_id] : -INFINITY;

        #pragma unroll
        for (int shuffle_offset = 16; shuffle_offset > 0; shuffle_offset /= 2) {
            thread_value = fmaxf(thread_value, __shfl_down_sync(0xffffffff, thread_value, shuffle_offset));
        }

        if (thread_id == 0) {
            shared_memory[0] = thread_value;
        }
    }

    __syncthreads();

    return shared_memory[0];
}

__device__ __forceinline__ float block_reduce_sum(float thread_value, int thread_id, float* shared_memory) {
    #pragma unroll
    for (int shuffle_offset = 16; shuffle_offset > 0; shuffle_offset /= 2) {
        thread_value += __shfl_down_sync(0xffffffff, thread_value, shuffle_offset);
    }

    if (thread_id % WARP_SIZE == 0) {
        shared_memory[thread_id / WARP_SIZE] = thread_value;
    }

    __syncthreads();

    if (thread_id < WARP_SIZE) {
        thread_value = (thread_id < WARPS_PER_BLOCK) ? shared_memory[thread_id] : 0.0f;

        #pragma unroll
        for (int shuffle_offset = 16; shuffle_offset > 0; shuffle_offset /= 2) {
            thread_value += __shfl_down_sync(0xffffffff, thread_value, shuffle_offset);
        }

        if (thread_id == 0) {
            shared_memory[0] = thread_value;
        }
    }

    __syncthreads();

    return shared_memory[0];
}

__global__ void softmax_kernel(const float* __restrict__ input, float* __restrict__ output, int elements_per_row) {
    const int row_index = blockIdx.x;
    const int thread_id = threadIdx.x;

    const float* input_row = input + row_index * elements_per_row;
    float* output_row = output + row_index * elements_per_row;

    __shared__ float shared_memory[WARPS_PER_BLOCK];

    float thread_max = -INFINITY;
    const int vector_size = elements_per_row / 4;
    const float4* input_row_vec4 = reinterpret_cast<const float4*>(input_row);

    for (int i = thread_id; i < vector_size; i += blockDim.x) {
        float4 values = __ldg(&input_row_vec4[i]);
        thread_max = fmaxf(thread_max, fmaxf(fmaxf(values.x, values.y), fmaxf(values.z, values.w)));
    }
    
    float row_max = block_reduce_max(thread_max, thread_id, shared_memory);

    float thread_sum = 0.0f;
    float4* output_row_vec4 = reinterpret_cast<float4*>(output_row);

    for (int i = thread_id; i < vector_size; i += blockDim.x) {
        float4 values = __ldg(&input_row_vec4[i]);
        values.x = __expf(values.x - row_max);
        values.y = __expf(values.y - row_max);
        values.z = __expf(values.z - row_max);
        values.w = __expf(values.w - row_max);
        output_row_vec4[i] = values;
        thread_sum += (values.x + values.y + values.z + values.w);
    }
    
    float row_sum = block_reduce_sum(thread_sum, thread_id, shared_memory);
    float inverse_sum = 1.0f / row_sum;

    for (int i = thread_id; i < vector_size; i += blockDim.x) {
        float4 values = output_row_vec4[i];
        values.x *= inverse_sum;
        values.y *= inverse_sum;
        values.z *= inverse_sum;
        values.w *= inverse_sum;
        output_row_vec4[i] = values;
    }
}

struct Cache {
    float* device_input = nullptr;
    float* device_output = nullptr;
    float* pinned_input = nullptr;
    float* pinned_output = nullptr;
    std::size_t capacity = 0;
    cudaStream_t stream = nullptr;

    ~Cache() {
        if (device_input) cudaFree(device_input);
        if (device_output) cudaFree(device_output);
        if (pinned_input) cudaFreeHost(pinned_input);
        if (pinned_output) cudaFreeHost(pinned_output);
        if (stream) cudaStreamDestroy(stream);
    }

    void initialize(std::size_t element_count) {
        if (stream == nullptr) {
            CHECK_CUDA(cudaStreamCreate(&stream));
        }
        if (element_count > capacity) {
            if (device_input) cudaFree(device_input);
            if (device_output) cudaFree(device_output);
            if (pinned_input) cudaFreeHost(pinned_input);
            if (pinned_output) cudaFreeHost(pinned_output);

            const std::size_t total_bytes = element_count * sizeof(float);

            CHECK_CUDA(cudaMalloc(&device_input, total_bytes));
            CHECK_CUDA(cudaMalloc(&device_output, total_bytes));
            CHECK_CUDA(cudaMallocHost(&pinned_input, total_bytes));
            CHECK_CUDA(cudaMallocHost(&pinned_output, total_bytes));

            capacity = element_count;
        }
    }
};

Cache& get_cache() {
    static Cache instance;
    return instance;
}

} // namespace

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (input.empty() || row_count <= 0) {
        return {};
    }

    const std::size_t total_elements = input.size();
    const int elements_per_row = static_cast<int>(total_elements / row_count);
    const std::size_t total_bytes = total_elements * sizeof(float);

    Cache& cache = get_cache();
    cache.initialize(total_elements);

    std::memcpy(cache.pinned_input, input.data(), total_bytes);
    CHECK_CUDA(cudaMemcpyAsync(cache.device_input,
                               cache.pinned_input,
                               total_bytes,
                               cudaMemcpyHostToDevice,
                               cache.stream));

    softmax_kernel<<<row_count, BLOCK_THREADS, 0, cache.stream>>>(cache.device_input,
                                                                  cache.device_output,
                                                                  elements_per_row);
    CHECK_CUDA(cudaGetLastError());

    std::vector<float> output(total_elements);

    CHECK_CUDA(cudaMemcpyAsync(cache.pinned_output,
                               cache.device_output,
                               total_bytes,
                               cudaMemcpyDeviceToHost,
                               cache.stream));
    CHECK_CUDA(cudaStreamSynchronize(cache.stream));

    std::memcpy(output.data(), cache.pinned_output, total_bytes);

    return output;
}
