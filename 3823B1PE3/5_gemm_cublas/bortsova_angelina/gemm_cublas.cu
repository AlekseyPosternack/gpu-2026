#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace {

__global__ void TransposeResult(const float* product, float* result, int matrix_size) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < matrix_size && column < matrix_size) {
        result[column * matrix_size + row] = product[row * matrix_size + column];
    }
}

struct CublasWorkspace {
    cublasHandle_t handle;
    cudaStream_t stream;
    float* left_values = nullptr;
    float* right_values = nullptr;
    float* product_values = nullptr;
    size_t capacity = 0;

    CublasWorkspace() {
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        cublasCreate(&handle);
        cublasSetStream(handle, stream);
        cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
    }

    ~CublasWorkspace() {
        cublasDestroy(handle);
        cudaFree(left_values);
        cudaFree(right_values);
        cudaFree(product_values);
        cudaStreamDestroy(stream);
    }

    void Reserve(size_t element_count) {
        if (element_count > capacity) {
            cudaFree(left_values);
            cudaFree(right_values);
            cudaFree(product_values);
            size_t byte_count = element_count * sizeof(float);
            cudaMalloc(&left_values, byte_count);
            cudaMalloc(&right_values, byte_count);
            cudaMalloc(&product_values, byte_count);
            capacity = element_count;
        }
    }
};

}

std::vector<float> GemmCUBLAS(const std::vector<float>& left_matrix, const std::vector<float>& right_matrix, int matrix_size) {
    if (left_matrix.empty()) {
        return {};
    }

    static CublasWorkspace workspace;
    size_t element_count = static_cast<size_t>(matrix_size) * matrix_size;
    size_t byte_count = element_count * sizeof(float);
    workspace.Reserve(element_count);
    cudaMemcpyAsync(workspace.left_values, left_matrix.data(), byte_count,
                    cudaMemcpyHostToDevice, workspace.stream);
    cudaMemcpyAsync(workspace.right_values, right_matrix.data(), byte_count,
                    cudaMemcpyHostToDevice, workspace.stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(workspace.handle, CUBLAS_OP_T, CUBLAS_OP_T,
                matrix_size, matrix_size, matrix_size, &alpha,
                workspace.left_values, matrix_size, workspace.right_values, matrix_size,
                &beta, workspace.product_values, matrix_size);

    dim3 threads(32, 8);
    dim3 blocks((matrix_size + threads.x - 1) / threads.x,
                (matrix_size + threads.y - 1) / threads.y);
    TransposeResult<<<blocks, threads, 0, workspace.stream>>>(
        workspace.product_values, workspace.left_values, matrix_size);

    std::vector<float> result(element_count);
    cudaMemcpyAsync(result.data(), workspace.left_values, byte_count,
                    cudaMemcpyDeviceToHost, workspace.stream);
    cudaStreamSynchronize(workspace.stream);
    return result;
}
