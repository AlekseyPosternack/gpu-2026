#include "softmax_cuda.h"

#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstddef>
#include <thread>

namespace {

constexpr int kBlockThreads = 512;
constexpr int kWarpCount = kBlockThreads / 32;
constexpr int kVecPerThread = 8;
constexpr int kMaxCachedRow = kBlockThreads * kVecPerThread * 4;

__device__ __forceinline__ float WarpReduceMax(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffffu, val, offset));
    }
    return val;
}

__device__ __forceinline__ float WarpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, offset);
    }
    return val;
}

__global__ void __launch_bounds__(kBlockThreads)
RowSoftmaxKernel(float* __restrict__ data, int row_size) {
    __shared__ float s_max_warps[kWarpCount];
    __shared__ float s_sum_warps[kWarpCount];
    __shared__ float s_row_max;
    __shared__ float s_inv_sum;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;

    float* const row_ptr = data + static_cast<std::size_t>(blockIdx.x) * row_size;

    if ((row_size & 3) == 0 && row_size <= kMaxCachedRow) {
        const int vec_count = row_size >> 2;
        float4* const row_vec = reinterpret_cast<float4*>(row_ptr);
        float4 reg_cache[kVecPerThread];

        float local_max = -CUDART_INF_F;

        #pragma unroll
        for (int k = 0; k < kVecPerThread; ++k) {
            const int idx = tid + k * kBlockThreads;
            if (idx < vec_count) {
                const float4 v = row_vec[idx];
                reg_cache[k] = v;
                local_max = fmaxf(local_max, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
            }
        }

        local_max = WarpReduceMax(local_max);
        if (lane == 0) {
            s_max_warps[warp_id] = local_max;
        }
        __syncthreads();

        if (warp_id == 0) {
            float val = (lane < kWarpCount) ? s_max_warps[lane] : -CUDART_INF_F;
            #pragma unroll
            for (int offset = 8; offset > 0; offset >>= 1) {
                val = fmaxf(val, __shfl_down_sync(0xffffffffu, val, offset));
            }
            if (lane == 0) {
                s_row_max = val;
            }
        }
        __syncthreads();

        const float row_max = s_row_max;
        float local_sum = 0.0f;

        #pragma unroll
        for (int k = 0; k < kVecPerThread; ++k) {
            const int idx = tid + k * kBlockThreads;
            if (idx < vec_count) {
                reg_cache[k].x = __expf(reg_cache[k].x - row_max);
                reg_cache[k].y = __expf(reg_cache[k].y - row_max);
                reg_cache[k].z = __expf(reg_cache[k].z - row_max);
                reg_cache[k].w = __expf(reg_cache[k].w - row_max);
                local_sum += (reg_cache[k].x + reg_cache[k].y) + (reg_cache[k].z + reg_cache[k].w);
            }
        }

        local_sum = WarpReduceSum(local_sum);
        if (lane == 0) {
            s_sum_warps[warp_id] = local_sum;
        }
        __syncthreads();

        if (warp_id == 0) {
            float val = (lane < kWarpCount) ? s_sum_warps[lane] : 0.0f;
            #pragma unroll
            for (int offset = 8; offset > 0; offset >>= 1) {
                val += __shfl_down_sync(0xffffffffu, val, offset);
            }
            if (lane == 0) {
                s_inv_sum = 1.0f / val;
            }
        }
        __syncthreads();

        const float inv_sum = s_inv_sum;
        #pragma unroll
        for (int k = 0; k < kVecPerThread; ++k) {
            const int idx = tid + k * kBlockThreads;
            if (idx < vec_count) {
                reg_cache[k].x *= inv_sum;
                reg_cache[k].y *= inv_sum;
                reg_cache[k].z *= inv_sum;
                reg_cache[k].w *= inv_sum;
                row_vec[idx] = reg_cache[k];
            }
        }
    } else {
        float local_max = -CUDART_INF_F;
        for (int col = tid; col < row_size; col += kBlockThreads) {
            local_max = fmaxf(local_max, row_ptr[col]);
        }

        local_max = WarpReduceMax(local_max);
        if (lane == 0) {
            s_max_warps[warp_id] = local_max;
        }
        __syncthreads();

        if (warp_id == 0) {
            float val = (lane < kWarpCount) ? s_max_warps[lane] : -CUDART_INF_F;
            #pragma unroll
            for (int offset = 8; offset > 0; offset >>= 1) {
                val = fmaxf(val, __shfl_down_sync(0xffffffffu, val, offset));
            }
            if (lane == 0) {
                s_row_max = val;
            }
        }
        __syncthreads();

        const float row_max = s_row_max;
        float local_sum = 0.0f;
        for (int col = tid; col < row_size; col += kBlockThreads) {
            const float e = __expf(row_ptr[col] - row_max);
            row_ptr[col] = e;
            local_sum += e;
        }

        local_sum = WarpReduceSum(local_sum);
        if (lane == 0) {
            s_sum_warps[warp_id] = local_sum;
        }
        __syncthreads();

        if (warp_id == 0) {
            float val = (lane < kWarpCount) ? s_sum_warps[lane] : 0.0f;
            #pragma unroll
            for (int offset = 8; offset > 0; offset >>= 1) {
                val += __shfl_down_sync(0xffffffffu, val, offset);
            }
            if (lane == 0) {
                s_inv_sum = 1.0f / val;
            }
        }
        __syncthreads();

        const float inv_sum = s_inv_sum;
        for (int col = tid; col < row_size; col += kBlockThreads) {
            row_ptr[col] *= inv_sum;
        }
    }
}

} // namespace

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    const std::size_t total_elems = input.size();
    if (total_elems == 0 || row_count <= 0) {
        return {};
    }

    const int row_size = static_cast<int>(total_elems / static_cast<std::size_t>(row_count));
    const std::size_t total_bytes = total_elems * sizeof(float);

    static thread_local float* dev_buf = nullptr;
    static thread_local std::size_t allocated_elems = 0;

    if (total_elems > allocated_elems) {
        if (dev_buf != nullptr) {
            cudaFree(dev_buf);
        }
        cudaMalloc(&dev_buf, total_bytes);
        allocated_elems = total_elems;
    }

    std::vector<float> result;
    std::thread host_alloc_worker([&result, total_elems]() {
        result.resize(total_elems);
    });

    cudaMemcpy(dev_buf, input.data(), total_bytes, cudaMemcpyHostToDevice);

    RowSoftmaxKernel<<<row_count, kBlockThreads>>>(dev_buf, row_size);

    host_alloc_worker.join();

    cudaMemcpy(result.data(), dev_buf, total_bytes, cudaMemcpyDeviceToHost);

    return result;
}
