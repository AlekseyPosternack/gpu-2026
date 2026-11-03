#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <vector>

namespace
{

    constexpr int BLOCK_THREADS = 256;
    constexpr float COEFF_A = -1.59576912f;        // -2 * sqrt(2/pi)
    constexpr float COEFF_B = COEFF_A * 0.044715f; // -0.07135482f

    __device__ __forceinline__ float compute_gelu(float x)
    {
        return __fdividef(x, 1.0f + __expf(x * (COEFF_A + COEFF_B * x * x)));
    }

    __global__ void GeluKernel(float *__restrict__ data, size_t n)
    {
        size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        size_t n_vec = n / 4;

        if (tid < n_vec)
        {
            float4 *vec_ptr = reinterpret_cast<float4 *>(data);
            float4 item = vec_ptr[tid];
            item.x = compute_gelu(item.x);
            item.y = compute_gelu(item.y);
            item.z = compute_gelu(item.z);
            item.w = compute_gelu(item.w);
            vec_ptr[tid] = item;
        }
        else
        {
            size_t rem_idx = n_vec * 4 + (tid - n_vec);
            if (rem_idx < n)
            {
                data[rem_idx] = compute_gelu(data[rem_idx]);
            }
        }
    }

    static float *g_device_mem = nullptr;
    static size_t g_allocated_elements = 0;
    static cudaStream_t g_stream = nullptr;

    void ensure_gpu_resources(size_t n)
    {
        if (!g_stream)
        {
            cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking);
        }
        if (n > g_allocated_elements)
        {
            if (g_device_mem)
            {
                cudaFree(g_device_mem);
            }
            cudaMalloc(&g_device_mem, n * sizeof(float));
            g_allocated_elements = n;
        }
    }

} // namespace

std::vector<float> GeluCUDA(const std::vector<float> &input)
{
    const size_t n = input.size();
    if (n == 0)
    {
        return {};
    }

    const size_t bytes = n * sizeof(float);
    ensure_gpu_resources(n);

    cudaHostRegister(const_cast<float *>(input.data()), bytes, cudaHostRegisterDefault);

    cudaMemcpyAsync(g_device_mem, input.data(), bytes, cudaMemcpyHostToDevice, g_stream);

    const size_t total_tasks = (n + 3) / 4;
    const unsigned int grid_blocks = static_cast<unsigned int>((total_tasks + BLOCK_THREADS - 1) / BLOCK_THREADS);
    GeluKernel<<<grid_blocks, BLOCK_THREADS, 0, g_stream>>>(g_device_mem, n);

    std::vector<float> output(n);

    cudaHostRegister(output.data(), bytes, cudaHostRegisterDefault);
    cudaMemcpyAsync(output.data(), g_device_mem, bytes, cudaMemcpyDeviceToHost, g_stream);

    cudaStreamSynchronize(g_stream);
    cudaHostUnregister(const_cast<float *>(input.data()));
    cudaHostUnregister(output.data());

    return output;
}
