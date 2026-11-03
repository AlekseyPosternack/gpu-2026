#include "softmax_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstddef>
#include <cmath>

template <int BLOCK_THREADS, int VEC_ITEMS>
__global__ void softmax_reg_kernel(const float* __restrict__ input,
                                   float* __restrict__ output) {
    constexpr int ROW_SIZE = BLOCK_THREADS * VEC_ITEMS * 4;
    const size_t row = blockIdx.x;
    const int tid = threadIdx.x;

    const float4* row_in = reinterpret_cast<const float4*>(input + row * ROW_SIZE);
    float4* row_out = reinterpret_cast<float4*>(output + row * ROW_SIZE);

    float4 reg[VEC_ITEMS];
    float local_max = -INFINITY;

    #pragma unroll
    for (int k = 0; k < VEC_ITEMS; ++k) {
        reg[k] = row_in[k * BLOCK_THREADS + tid];
        local_max = fmaxf(local_max, fmaxf(fmaxf(reg[k].x, reg[k].y), fmaxf(reg[k].z, reg[k].w)));
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }

    __shared__ float s_val[32];
    __shared__ float s_broadcast;
    const int lane = tid & 31;
    const int wid = tid >> 5;

    if (lane == 0) {
        s_val[wid] = local_max;
    }
    __syncthreads();

    if (wid == 0) {
        float val = (lane < (BLOCK_THREADS / 32)) ? s_val[lane] : -INFINITY;
        #pragma unroll
        for (int offset = (BLOCK_THREADS / 64); offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        }
        if (lane == 0) {
            s_broadcast = val;
        }
    }
    __syncthreads();
    const float row_max = s_broadcast;

    float local_sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < VEC_ITEMS; ++k) {
        reg[k].x = __expf(reg[k].x - row_max);
        reg[k].y = __expf(reg[k].y - row_max);
        reg[k].z = __expf(reg[k].z - row_max);
        reg[k].w = __expf(reg[k].w - row_max);
        local_sum += (reg[k].x + reg[k].y) + (reg[k].z + reg[k].w);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    if (lane == 0) {
        s_val[wid] = local_sum;
    }
    __syncthreads();

    if (wid == 0) {
        float val = (lane < (BLOCK_THREADS / 32)) ? s_val[lane] : 0.0f;
        #pragma unroll
        for (int offset = (BLOCK_THREADS / 64); offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0) {
            s_broadcast = 1.0f / val;
        }
    }
    __syncthreads();
    const float inv_sum = s_broadcast;

    #pragma unroll
    for (int k = 0; k < VEC_ITEMS; ++k) {
        reg[k].x *= inv_sum;
        reg[k].y *= inv_sum;
        reg[k].z *= inv_sum;
        reg[k].w *= inv_sum;
        row_out[k * BLOCK_THREADS + tid] = reg[k];
    }
}

template <int BLOCK_THREADS>
__global__ void softmax_general_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int row_size) {
    const size_t row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* row_in = input + row * static_cast<size_t>(row_size);
    float* row_out = output + row * static_cast<size_t>(row_size);

    float local_max = -INFINITY;
    for (int i = tid; i < row_size; i += BLOCK_THREADS) {
        local_max = fmaxf(local_max, row_in[i]);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }

    __shared__ float s_val[32];
    __shared__ float s_broadcast;
    const int lane = tid & 31;
    const int wid = tid >> 5;

    if (lane == 0) {
        s_val[wid] = local_max;
    }
    __syncthreads();

    if (wid == 0) {
        float val = (lane < (BLOCK_THREADS / 32)) ? s_val[lane] : -INFINITY;
        #pragma unroll
        for (int offset = (BLOCK_THREADS / 64); offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        }
        if (lane == 0) {
            s_broadcast = val;
        }
    }
    __syncthreads();
    const float row_max = s_broadcast;

    float local_sum = 0.0f;
    for (int i = tid; i < row_size; i += BLOCK_THREADS) {
        float e = __expf(row_in[i] - row_max);
        row_out[i] = e;
        local_sum += e;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    if (lane == 0) {
        s_val[wid] = local_sum;
    }
    __syncthreads();

    if (wid == 0) {
        float val = (lane < (BLOCK_THREADS / 32)) ? s_val[lane] : 0.0f;
        #pragma unroll
        for (int offset = (BLOCK_THREADS / 64); offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0) {
            s_broadcast = 1.0f / val;
        }
    }
    __syncthreads();
    const float inv_sum = s_broadcast;

    for (int i = tid; i < row_size; i += BLOCK_THREADS) {
        row_out[i] *= inv_sum;
    }
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    size_t total_elems = input.size();
    if (total_elems == 0 || row_count <= 0) {
        return {};
    }

    int row_size = static_cast<int>(total_elems / row_count);
    size_t bytes = total_elems * sizeof(float);

    static float* d_in = nullptr;
    static float* d_out = nullptr;
    static size_t cached_bytes = 0;
    static cudaStream_t stream;
    static bool initialized = false;

    static const float* registered_ptr = nullptr;
    static size_t registered_bytes = 0;

    if (!initialized) {
        cudaStreamCreate(&stream);
        initialized = true;
    }

    if (cached_bytes < bytes) {
        if (d_in) cudaFree(d_in);
        if (d_out) cudaFree(d_out);
        cudaMalloc(&d_in, bytes);
        cudaMalloc(&d_out, bytes);
        cached_bytes = bytes;
    }

    if (bytes >= (1 << 20)) {
        if (registered_ptr != input.data() || registered_bytes != bytes) {
            if (registered_ptr != nullptr) {
                cudaHostUnregister(const_cast<float*>(registered_ptr));
                cudaGetLastError();
            }
            if (cudaHostRegister(const_cast<float*>(input.data()), bytes, cudaHostRegisterDefault) == cudaSuccess) {
                registered_ptr = input.data();
                registered_bytes = bytes;
            } else {
                cudaGetLastError();
                registered_ptr = nullptr;
                registered_bytes = 0;
            }
        }
    }

    cudaMemcpyAsync(d_in, input.data(), bytes, cudaMemcpyHostToDevice, stream);

    if (row_size == 16384) {
        softmax_reg_kernel<512, 8><<<row_count, 512, 0, stream>>>(d_in, d_out);
    } else if (row_size == 8192) {
        softmax_reg_kernel<256, 8><<<row_count, 256, 0, stream>>>(d_in, d_out);
    } else if (row_size == 4096) {
        softmax_reg_kernel<256, 4><<<row_count, 256, 0, stream>>>(d_in, d_out);
    } else if (row_size == 2048) {
        softmax_reg_kernel<256, 2><<<row_count, 256, 0, stream>>>(d_in, d_out);
    } else if (row_size == 1024) {
        softmax_reg_kernel<256, 1><<<row_count, 256, 0, stream>>>(d_in, d_out);
    } else {
        softmax_general_kernel<256><<<row_count, 256, 0, stream>>>(d_in, d_out, row_size);
    }

    std::vector<float> output(total_elems);

    cudaMemcpyAsync(output.data(), d_out, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    return output;
}