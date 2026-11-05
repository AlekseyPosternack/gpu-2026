#include "softmax_cuda.h"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <future>

namespace {

constexpr int kBlock = 256;
constexpr int kWarps = kBlock / 32;
constexpr int kMaxVecs = 16;

struct MaxSum {
    float m;
    float s;
};

//------------------------------------------------------------------
//function: Combine
//purpose: объединяет две пары (максимум, сумма экспонент) в одну, пересчитывая сумму к большему максимуму
//------------------------------------------------------------------
__device__ __forceinline__ MaxSum Combine(MaxSum a, MaxSum b) {
    if (a.m == -CUDART_INF_F) return b;
    if (b.m == -CUDART_INF_F) return a;
    if (a.m < b.m) {
        const MaxSum t = a;
        a = b;
        b = t;
    }
    a.s += b.s * __expf(b.m - a.m);
    return a;
}

//------------------------------------------------------------------
//function: WarpReduce
//purpose: сводит пары (максимум, сумма) 32 потоков варпа в одну через обмен регистрами
//------------------------------------------------------------------
__device__ __forceinline__ MaxSum WarpReduce(MaxSum v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        MaxSum other;
        other.m = __shfl_xor_sync(0xffffffff, v.m, offset);
        other.s = __shfl_xor_sync(0xffffffff, v.s, offset);
        v = Combine(v, other);
    }
    return v;
}

//------------------------------------------------------------------
//function: BlockReduce
//purpose: находит максимум строки и сумму экспонент по всему блоку и раздаёт результат всем потокам
//------------------------------------------------------------------
__device__ __forceinline__ MaxSum BlockReduce(MaxSum v, MaxSum* shared) {
    v = WarpReduce(v);
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (lane == 0) shared[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = lane < kWarps ? shared[lane] : MaxSum{-CUDART_INF_F, 0.0f};
        v = WarpReduce(v);
        if (lane == 0) shared[0] = v;
    }
    __syncthreads();
    return shared[0];
}

//------------------------------------------------------------------
//function: SoftmaxCachedKernel
//purpose: считает softmax одной строки на блок, читая строку из памяти один раз и держа её в регистрах
//------------------------------------------------------------------
template <int kVecs>
__global__ void __launch_bounds__(kBlock) SoftmaxCachedKernel(float* data, int row_size) {
    __shared__ MaxSum shared[kWarps];
    float4* row = reinterpret_cast<float4*>(data + static_cast<size_t>(blockIdx.x) * row_size);
    const int vecs = row_size / 4;

    float4 v[kVecs];
    float local_max = -CUDART_INF_F;
#pragma unroll
    for (int i = 0; i < kVecs; ++i) {
        const int j = threadIdx.x + i * kBlock;
        v[i] = j < vecs ? row[j]
                        : make_float4(-CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F);
        local_max = fmaxf(local_max, fmaxf(fmaxf(v[i].x, v[i].y), fmaxf(v[i].z, v[i].w)));
    }
    float local_sum = 0.0f;
    if (local_max != -CUDART_INF_F) {
#pragma unroll
        for (int i = 0; i < kVecs; ++i) {
            v[i].x = __expf(v[i].x - local_max);
            v[i].y = __expf(v[i].y - local_max);
            v[i].z = __expf(v[i].z - local_max);
            v[i].w = __expf(v[i].w - local_max);
            local_sum += (v[i].x + v[i].y) + (v[i].z + v[i].w);
        }
    }
    const MaxSum total = BlockReduce(MaxSum{local_max, local_sum}, shared);
    const float scale = local_max == -CUDART_INF_F ? 0.0f : __expf(local_max - total.m) / total.s;
#pragma unroll
    for (int i = 0; i < kVecs; ++i) {
        const int j = threadIdx.x + i * kBlock;
        if (j < vecs) {
            row[j] = make_float4(v[i].x * scale, v[i].y * scale, v[i].z * scale, v[i].w * scale);
        }
    }
}

//------------------------------------------------------------------
//function: SoftmaxGenericKernel
//purpose: запасной вариант softmax для строк, которые не помещаются в регистры или не кратны 4
//------------------------------------------------------------------
__global__ void __launch_bounds__(kBlock) SoftmaxGenericKernel(float* data, int row_size) {
    __shared__ MaxSum shared[kWarps];
    float* row = data + static_cast<size_t>(blockIdx.x) * row_size;

    MaxSum local = {-CUDART_INF_F, 0.0f};
    for (int i = threadIdx.x; i < row_size; i += kBlock) {
        local = Combine(local, MaxSum{row[i], 1.0f});
    }
    const MaxSum total = BlockReduce(local, shared);
    const float inv_sum = 1.0f / total.s;
    for (int i = threadIdx.x; i < row_size; i += kBlock) {
        row[i] = __expf(row[i] - total.m) * inv_sum;
    }
}

//------------------------------------------------------------------
//function: Launch
//purpose: выбирает ядро под длину строки, чтобы каждый поток держал в регистрах не больше нужного
//------------------------------------------------------------------
void Launch(float* data, int rows, int row_size) {
    const int vecs = row_size / 4;
    if (row_size % 4 != 0 || vecs > kMaxVecs * kBlock) {
        SoftmaxGenericKernel<<<rows, kBlock>>>(data, row_size);
    } else if (vecs <= kBlock) {
        SoftmaxCachedKernel<1><<<rows, kBlock>>>(data, row_size);
    } else if (vecs <= 2 * kBlock) {
        SoftmaxCachedKernel<2><<<rows, kBlock>>>(data, row_size);
    } else if (vecs <= 4 * kBlock) {
        SoftmaxCachedKernel<4><<<rows, kBlock>>>(data, row_size);
    } else if (vecs <= 8 * kBlock) {
        SoftmaxCachedKernel<8><<<rows, kBlock>>>(data, row_size);
    } else {
        SoftmaxCachedKernel<16><<<rows, kBlock>>>(data, row_size);
    }
}

float* g_buffer = nullptr;
size_t g_capacity = 0;

//------------------------------------------------------------------
//function: EnsureCapacity
//purpose: выделяет видеопамять только при росте размера, чтобы не вызывать cudaMalloc на каждом запуске
//------------------------------------------------------------------
void EnsureCapacity(size_t size) {
    if (size <= g_capacity) return;
    if (g_buffer) cudaFree(g_buffer);
    cudaMalloc(&g_buffer, size * sizeof(float));
    g_capacity = size;
}

}

//------------------------------------------------------------------
//function: SoftmaxCUDA
//purpose: считает softmax каждой строки матрицы на GPU, пока CPU параллельно выделяет память под результат
//------------------------------------------------------------------
std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    const size_t size = input.size();
    if (size == 0 || row_count <= 0) return {};
    const int row_size = static_cast<int>(size / row_count);
    const size_t bytes = size * sizeof(float);

    auto result = std::async(std::launch::async,
                             [size] { return std::vector<float>(size); });

    EnsureCapacity(size);
    cudaMemcpyAsync(g_buffer, input.data(), bytes, cudaMemcpyHostToDevice);
    Launch(g_buffer, row_count, row_size);

    std::vector<float> output = result.get();
    cudaMemcpy(output.data(), g_buffer, bytes, cudaMemcpyDeviceToHost);
    return output;
}
