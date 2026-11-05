#include "gemm_cublas.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <future>

namespace {

constexpr int kStrips = 4;

struct Context {
    bool initialized = false;
    cublasHandle_t handles[kStrips];
    cudaStream_t streams[kStrips];
    cudaEvent_t b_ready;
    float* buffer = nullptr;
    size_t capacity = 0;
};

Context g_ctx;

//------------------------------------------------------------------
//function: Prepare
//purpose: один раз создаёт CUDA-потоки и cuBLAS-хендлы и выделяет видеопамять под A, B и C, переиспользуя её между вызовами
//------------------------------------------------------------------
void Prepare(size_t elements) {
    if (!g_ctx.initialized) {
        for (int i = 0; i < kStrips; ++i) {
            cudaStreamCreateWithFlags(&g_ctx.streams[i], cudaStreamNonBlocking);
            cublasCreate(&g_ctx.handles[i]);
            cublasSetStream(g_ctx.handles[i], g_ctx.streams[i]);
        }
        cudaEventCreateWithFlags(&g_ctx.b_ready, cudaEventDisableTiming);
        g_ctx.initialized = true;
    }
    if (elements <= g_ctx.capacity) return;
    if (g_ctx.buffer) cudaFree(g_ctx.buffer);
    cudaMalloc(&g_ctx.buffer, 3 * elements * sizeof(float));
    g_ctx.capacity = elements;
}

}

//------------------------------------------------------------------
//function: GemmCUBLAS
//purpose: умножает матрицы A и B (n x n, по строкам) через cuBLAS, по полосам строк, чтобы копирование и вычисления шли одновременно
//------------------------------------------------------------------
std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (n <= 0) return {};
    const size_t elements = static_cast<size_t>(n) * n;

    auto result = std::async(std::launch::async,
                             [elements] { return std::vector<float>(elements); });

    Prepare(elements);
    float* d_a = g_ctx.buffer;
    float* d_b = g_ctx.buffer + elements;
    float* d_c = g_ctx.buffer + 2 * elements;

    const int strips = n % kStrips == 0 && n >= 64 ? kStrips : 1;
    const int rows = n / strips;
    const size_t strip_elements = static_cast<size_t>(rows) * n;
    const size_t strip_bytes = strip_elements * sizeof(float);

    cudaMemcpyAsync(d_b, b.data(), elements * sizeof(float),
                    cudaMemcpyHostToDevice, g_ctx.streams[0]);
    cudaEventRecord(g_ctx.b_ready, g_ctx.streams[0]);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    for (int s = 0; s < strips; ++s) {
        const size_t offset = s * strip_elements;
        cudaStream_t stream = g_ctx.streams[s];
        cudaMemcpyAsync(d_a + offset, a.data() + offset, strip_bytes,
                        cudaMemcpyHostToDevice, stream);
        cudaStreamWaitEvent(stream, g_ctx.b_ready, 0);
        cublasSgemm(g_ctx.handles[s], CUBLAS_OP_N, CUBLAS_OP_N, n, rows, n,
                    &alpha, d_b, n, d_a + offset, n, &beta, d_c + offset, n);
    }

    std::vector<float> c = result.get();
    for (int s = 0; s < strips; ++s) {
        const size_t offset = s * strip_elements;
        cudaMemcpyAsync(c.data() + offset, d_c + offset, strip_bytes,
                        cudaMemcpyDeviceToHost, g_ctx.streams[s]);
    }
    for (int s = 0; s < strips; ++s) cudaStreamSynchronize(g_ctx.streams[s]);
    return c;
}
