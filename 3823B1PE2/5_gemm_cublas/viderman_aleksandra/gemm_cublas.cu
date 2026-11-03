#include "gemm_cublas.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <thread>
#include <vector>

namespace {

constexpr int kNumStreams = 4;

struct CublasPipeline {
    cublasHandle_t handles[kNumStreams] = {};
    cudaStream_t streams[kNumStreams] = {};
    cudaStream_t upload_stream = nullptr;
    cudaEvent_t b_uploaded = nullptr;

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    std::size_t capacity = 0;

    CublasPipeline() {
        cudaStreamCreateWithFlags(&upload_stream, cudaStreamNonBlocking);
        cudaEventCreateWithFlags(&b_uploaded, cudaEventDisableTiming);

        for (int i = 0; i < kNumStreams; ++i) {
            cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
            cublasCreate(&handles[i]);
            cublasSetStream(handles[i], streams[i]);
            cublasSetPointerMode(handles[i], CUBLAS_POINTER_MODE_HOST);
        }
    }

    void Ensure(std::size_t count) {
        if (count > capacity) {
            if (d_a != nullptr) cudaFree(d_a);
            if (d_b != nullptr) cudaFree(d_b);
            if (d_c != nullptr) cudaFree(d_c);

            cudaMalloc(&d_a, count * sizeof(float));
            cudaMalloc(&d_b, count * sizeof(float));
            cudaMalloc(&d_c, count * sizeof(float));
            capacity = count;
        }
    }

    ~CublasPipeline() {
        for (int i = 0; i < kNumStreams; ++i) {
            if (streams[i] != nullptr) cudaStreamSynchronize(streams[i]);
            if (handles[i] != nullptr) cublasDestroy(handles[i]);
            if (streams[i] != nullptr) cudaStreamDestroy(streams[i]);
        }
        if (upload_stream != nullptr) {
            cudaStreamSynchronize(upload_stream);
            cudaStreamDestroy(upload_stream);
        }
        if (b_uploaded != nullptr) cudaEventDestroy(b_uploaded);

        if (d_a != nullptr) cudaFree(d_a);
        if (d_b != nullptr) cudaFree(d_b);
        if (d_c != nullptr) cudaFree(d_c);
    }
};

CublasPipeline g_pipeline;

}  // namespace

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (n <= 0) {
        return {};
    }

    const std::size_t count = static_cast<std::size_t>(n) * n;
    const std::size_t bytes = count * sizeof(float);

    g_pipeline.Ensure(count);

    std::vector<float> c;
    std::thread host_alloc([&c, count] { c.resize(count); });

    const float alpha = 1.0f;
    const float beta = 0.0f;

    if (n >= 1024 && (n % kNumStreams == 0)) {
        cudaMemcpyAsync(g_pipeline.d_b, b.data(), bytes, cudaMemcpyHostToDevice, g_pipeline.upload_stream);
        cudaEventRecord(g_pipeline.b_uploaded, g_pipeline.upload_stream);

        const int rows_per_chunk = n / kNumStreams;
        const std::size_t chunk_elements = static_cast<std::size_t>(rows_per_chunk) * n;
        const std::size_t chunk_bytes = chunk_elements * sizeof(float);

        for (int i = 0; i < kNumStreams; ++i) {
            const std::size_t offset = i * chunk_elements;
            cudaStream_t st = g_pipeline.streams[i];

            cudaMemcpyAsync(g_pipeline.d_a + offset, a.data() + offset, chunk_bytes,
                            cudaMemcpyHostToDevice, st);
            cudaStreamWaitEvent(st, g_pipeline.b_uploaded, 0);

            cublasSgemm(g_pipeline.handles[i], CUBLAS_OP_N, CUBLAS_OP_N,
                        n, rows_per_chunk, n,
                        &alpha,
                        g_pipeline.d_b, n,
                        g_pipeline.d_a + offset, n,
                        &beta,
                        g_pipeline.d_c + offset, n);
        }

        host_alloc.join();

        for (int i = 0; i < kNumStreams; ++i) {
            const std::size_t offset = i * chunk_elements;
            cudaMemcpyAsync(c.data() + offset, g_pipeline.d_c + offset, chunk_bytes,
                            cudaMemcpyDeviceToHost, g_pipeline.streams[i]);
        }

        for (int i = 0; i < kNumStreams; ++i) {
            cudaStreamSynchronize(g_pipeline.streams[i]);
        }
    } else {
        cudaStream_t st = g_pipeline.streams[0];
        cudaMemcpyAsync(g_pipeline.d_a, a.data(), bytes, cudaMemcpyHostToDevice, st);
        cudaMemcpyAsync(g_pipeline.d_b, b.data(), bytes, cudaMemcpyHostToDevice, st);

        cublasSgemm(g_pipeline.handles[0], CUBLAS_OP_N, CUBLAS_OP_N,
                    n, n, n,
                    &alpha,
                    g_pipeline.d_b, n,
                    g_pipeline.d_a, n,
                    &beta,
                    g_pipeline.d_c, n);

        host_alloc.join();

        cudaMemcpyAsync(c.data(), g_pipeline.d_c, bytes, cudaMemcpyDeviceToHost, st);
        cudaStreamSynchronize(st);
    }

    return c;
}
