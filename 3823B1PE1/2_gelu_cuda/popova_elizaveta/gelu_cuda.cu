#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstddef>

namespace
{

    constexpr float kScale = 1.5957691216f;
    constexpr float kCoef = 0.0713548163f;

    __global__ void GeluKernel(const float *__restrict__ in,
                               float *__restrict__ out,
                               int size)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= size)
            return;

        const float x = in[i];
        const float arg = -x * (kScale + kCoef * x * x);
        out[i] = x / (1.0f + __expf(arg));
    }

    struct DeviceBuffers
    {
        float *d_in = nullptr;
        float *d_out = nullptr;
        std::size_t cap = 0;

        ~DeviceBuffers()
        {
            if (d_in)
                cudaFree(d_in);
            if (d_out)
                cudaFree(d_out);
        }

        bool ensure(std::size_t bytes)
        {
            if (bytes <= cap)
                return true;
            if (d_in)
            {
                cudaFree(d_in);
                d_in = nullptr;
            }
            if (d_out)
            {
                cudaFree(d_out);
                d_out = nullptr;
            }
            if (cudaMalloc(&d_in, bytes) != cudaSuccess)
                return false;
            if (cudaMalloc(&d_out, bytes) != cudaSuccess)
            {
                cudaFree(d_in);
                d_in = nullptr;
                return false;
            }
            cap = bytes;
            return true;
        }
    };

    DeviceBuffers g_buf;

} // namespace

std::vector<float> GeluCUDA(const std::vector<float> &input)
{
    const std::size_t size = input.size();
    std::vector<float> output(size);
    if (size == 0)
        return output;

    const std::size_t bytes = size * sizeof(float);
    if (!g_buf.ensure(bytes))
    {
        std::fprintf(stderr, "CUDA: failed to allocate %zu bytes\n", bytes);
        return output;
    }

    cudaMemcpy(g_buf.d_in, input.data(), bytes, cudaMemcpyHostToDevice);

    constexpr int kBlock = 256;
    const int n = static_cast<int>(size);
    const int grid = (n + kBlock - 1) / kBlock;
    GeluKernel<<<grid, kBlock>>>(g_buf.d_in, g_buf.d_out, n);

    cudaMemcpy(output.data(), g_buf.d_out, bytes, cudaMemcpyDeviceToHost);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
    return output;
}