#include "softmax_cuda.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

namespace
{
    constexpr int BLOCK = 256;

    __device__ __forceinline__ float warp_max(float v)
    {
#pragma unroll
        for (int o = 16; o; o >>= 1)
            v = fmaxf(v, __shfl_down_sync(~0u, v, o));
        return v;
    }
    __device__ __forceinline__ float warp_sum(float v)
    {
#pragma unroll
        for (int o = 16; o; o >>= 1)
            v += __shfl_down_sync(~0u, v, o);
        return v;
    }

    __global__ void softmax_kernel(const float *__restrict__ in,
                                   float *__restrict__ out, int rs)
    {
        const int r = blockIdx.x, t = threadIdx.x, lane = t & 31, wid = t >> 5;
        const float *xin = in + (size_t)r * rs;
        float *xout = out + (size_t)r * rs;
        __shared__ float smem[BLOCK / 32];

        float m = -INFINITY;
        for (int i = t; i < rs; i += BLOCK)
            m = fmaxf(m, xin[i]);
        m = warp_max(m);
        if (!lane)
            smem[wid] = m;
        __syncthreads();
        if (!wid)
        {
            m = (t < BLOCK / 32) ? smem[t] : -INFINITY;
            m = warp_max(m);
            if (!t)
                smem[0] = m;
        }
        __syncthreads();
        const float M = smem[0];

        float s = 0.f;
        for (int i = t; i < rs; i += BLOCK)
            s += __expf(xin[i] - M);
        s = warp_sum(s);
        if (!lane)
            smem[wid] = s;
        __syncthreads();
        if (!wid)
        {
            s = (t < BLOCK / 32) ? smem[t] : 0.f;
            s = warp_sum(s);
            if (!t)
                smem[0] = s;
        }
        __syncthreads();
        const float inv = __fdividef(1.f, smem[0]);

        for (int i = t; i < rs; i += BLOCK)
            xout[i] = __expf(xin[i] - M) * inv;
    }

    float *d_in = nullptr, *d_out = nullptr, *h_in = nullptr, *h_out = nullptr;
    size_t g_cap = 0;

    void ensure(size_t bytes)
    {
        if (bytes <= g_cap)
            return;
        if (d_in)
        {
            cudaFree(d_in);
            cudaFree(d_out);
            cudaFreeHost(h_in);
            cudaFreeHost(h_out);
        }
        cudaMalloc(&d_in, bytes);
        cudaMalloc(&d_out, bytes);
        cudaMallocHost(&h_in, bytes);
        cudaMallocHost(&h_out, bytes);
        g_cap = bytes;
    }
} // namespace

std::vector<float> SoftmaxCUDA(const std::vector<float> &input, int rc)
{
    std::vector<float> out(input.size());
    if (input.empty() || rc <= 0)
        return out;
    int rs = (int)(input.size() / rc);
    size_t bytes = input.size() * sizeof(float);

    ensure(bytes);
    std::memcpy(h_in, input.data(), bytes);
    cudaMemcpyAsync(d_in, h_in, bytes, cudaMemcpyHostToDevice, 0);
    softmax_kernel<<<rc, BLOCK>>>(d_in, d_out, rs);
    cudaMemcpyAsync(h_out, d_out, bytes, cudaMemcpyDeviceToHost, 0);
    cudaDeviceSynchronize();
    std::memcpy(out.data(), h_out, bytes);

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e));
    return out;
}