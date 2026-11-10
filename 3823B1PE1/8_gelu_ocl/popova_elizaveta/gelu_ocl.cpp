#include "gelu_ocl.h"

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <cstdio>
#include <cstring>
#include <cstddef>

namespace
{

    const char *KERNEL_SRC = R"(
__kernel void gelu(__global const float* in, __global float* out, int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    float x = in[i];
    float arg = -x * (1.5957691216f + 0.0713548163f * x * x);
    out[i] = x / (1.0f + exp(arg));
}
)";

    cl_context g_ctx = nullptr;
    cl_command_queue g_queue = nullptr;
    cl_program g_program = nullptr;
    cl_kernel g_kernel = nullptr;
    cl_mem g_in = nullptr, g_out = nullptr;
    size_t g_cap = 0;
    int g_cur_platform = -1;

    void cleanup()
    {
        if (g_in)
            clReleaseMemObject(g_in);
        if (g_out)
            clReleaseMemObject(g_out);
        if (g_kernel)
            clReleaseKernel(g_kernel);
        if (g_program)
            clReleaseProgram(g_program);
        if (g_queue)
            clReleaseCommandQueue(g_queue);
        if (g_ctx)
            clReleaseContext(g_ctx);
        g_in = nullptr;
        g_out = nullptr;
        g_kernel = nullptr;
        g_program = nullptr;
        g_queue = nullptr;
        g_ctx = nullptr;
        g_cap = 0;
        g_cur_platform = -1;
    }

    bool init(int platform)
    {
        if (platform == g_cur_platform)
            return true;
        cleanup();

        cl_uint np = 0;
        if (clGetPlatformIDs(0, nullptr, &np) != CL_SUCCESS || np == 0)
            return false;
        if (platform < 0 || (cl_uint)platform >= np)
            return false;
        cl_platform_id plats[16];
        if (np > 16)
            np = 16;
        clGetPlatformIDs(np, plats, nullptr);

        cl_device_id dev;
        if (clGetDeviceIDs(plats[platform], CL_DEVICE_TYPE_GPU, 1, &dev, nullptr) != CL_SUCCESS)
            return false;

        cl_int err;
        g_ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
        if (err)
            return false;
        g_queue = clCreateCommandQueue(g_ctx, dev, 0, &err);
        if (err)
            return false;

        const char *src = KERNEL_SRC;
        size_t len = std::strlen(src);
        g_program = clCreateProgramWithSource(g_ctx, 1, &src, &len, &err);
        if (err)
            return false;

        if (clBuildProgram(g_program, 1, &dev, "-cl-fast-relaxed-math", nullptr, nullptr) != CL_SUCCESS)
        {
            char log[4096] = {0};
            size_t sz;
            clGetProgramBuildInfo(g_program, dev, CL_PROGRAM_BUILD_LOG, sizeof(log), log, &sz);
            return false;
        }
        g_kernel = clCreateKernel(g_program, "gelu", &err);
        if (err)
            return false;

        g_cur_platform = platform;
        return true;
    }

    bool ensure_buffers(size_t bytes)
    {
        if (bytes <= g_cap)
            return true;
        if (g_in)
        {
            clReleaseMemObject(g_in);
            g_in = nullptr;
        }
        if (g_out)
        {
            clReleaseMemObject(g_out);
            g_out = nullptr;
        }
        cl_int err;
        g_in = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY, bytes, nullptr, &err);
        if (err)
            return false;
        g_out = clCreateBuffer(g_ctx, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
        if (err)
            return false;
        g_cap = bytes;
        return true;
    }

} // namespace

std::vector<float> GeluOCL(const std::vector<float> &input, int platform)
{
    std::vector<float> output(input.size());
    if (input.empty())
        return output;

    if (!init(platform))
    {
        return output;
    }

    const size_t n = input.size();
    const size_t bytes = n * sizeof(float);
    if (!ensure_buffers(bytes))
    {

        return output;
    }

    clEnqueueWriteBuffer(g_queue, g_in, CL_TRUE, 0, bytes, input.data(), 0, nullptr, nullptr);

    int ni = static_cast<int>(n);
    clSetKernelArg(g_kernel, 0, sizeof(cl_mem), &g_in);
    clSetKernelArg(g_kernel, 1, sizeof(cl_mem), &g_out);
    clSetKernelArg(g_kernel, 2, sizeof(int), &ni);

    const size_t local = 256;
    const size_t global = ((n + local - 1) / local) * local;
    clEnqueueNDRangeKernel(g_queue, g_kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);
    clEnqueueReadBuffer(g_queue, g_out, CL_TRUE, 0, bytes, output.data(), 0, nullptr, nullptr);

    return output;
}