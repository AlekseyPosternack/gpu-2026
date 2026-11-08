#include "gelu_ocl.h"

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <vector>
#include <cstddef>

static const char* GELU_KERNEL_SRC = R"(
__kernel void gelu(__global const float* __restrict in,
                   __global float* __restrict out,
                   const int n)
{
    int i = (int)get_global_id(0);
    if (i < n) {
        float x = in[i];
        float z = 0.79788456f * (x + 0.044715f * x * x * x);
        out[i] = x / (1.0f + exp(-2.0f * z));
    }
}
)";

struct OCLState {
    int platformIdx = -1;
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_mem inBuf = nullptr;
    cl_mem outBuf = nullptr;
    size_t bufSize = 0;
};

static void releaseState(OCLState& st)
{
    if (st.inBuf) clReleaseMemObject(st.inBuf);
    if (st.outBuf) clReleaseMemObject(st.outBuf);
    if (st.kernel) clReleaseKernel(st.kernel);
    if (st.program) clReleaseProgram(st.program);
    if (st.queue) clReleaseCommandQueue(st.queue);
    if (st.context) clReleaseContext(st.context);
    st = OCLState();
}

static void initState(OCLState& st, int platformIdx)
{
    cl_uint numPlatforms = 0;
    clGetPlatformIDs(0, nullptr, &numPlatforms);
    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    st.platform = platforms[platformIdx];

    cl_uint numDevices = 0;
    clGetDeviceIDs(st.platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    std::vector<cl_device_id> devices(numDevices);
    clGetDeviceIDs(st.platform, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);
    st.device = devices[0];

    st.context = clCreateContext(nullptr, 1, &st.device, nullptr, nullptr, nullptr);
    st.queue = clCreateCommandQueue(st.context, st.device, 0, nullptr);
    st.program = clCreateProgramWithSource(st.context, 1, &GELU_KERNEL_SRC, nullptr, nullptr);
    clBuildProgram(st.program, 1, &st.device, nullptr, nullptr, nullptr);
    st.kernel = clCreateKernel(st.program, "gelu", nullptr);
    st.platformIdx = platformIdx;
}

std::vector<float> GeluOCL(const std::vector<float>& input, int platform)
{
    static OCLState st;
    if (st.platformIdx != platform) {
        if (st.platformIdx != -1) releaseState(st);
        initState(st, platform);
    }

    size_t n = input.size();
    if (n == 0) return {};

    size_t bytes = n * sizeof(float);
    if (st.bufSize < bytes) {
        if (st.inBuf) clReleaseMemObject(st.inBuf);
        if (st.outBuf) clReleaseMemObject(st.outBuf);
        st.inBuf = clCreateBuffer(st.context, CL_MEM_READ_ONLY, bytes, nullptr, nullptr);
        st.outBuf = clCreateBuffer(st.context, CL_MEM_WRITE_ONLY, bytes, nullptr, nullptr);
        st.bufSize = bytes;
    }

    clEnqueueWriteBuffer(st.queue, st.inBuf, CL_FALSE, 0, bytes,
                         input.data(), 0, nullptr, nullptr);

    clSetKernelArg(st.kernel, 0, sizeof(cl_mem), &st.inBuf);
    clSetKernelArg(st.kernel, 1, sizeof(cl_mem), &st.outBuf);
    int ni = (int)n;
    clSetKernelArg(st.kernel, 2, sizeof(int), &ni);

    size_t localSize = 256;
    size_t globalSize = ((n + localSize - 1) / localSize) * localSize;
    clEnqueueNDRangeKernel(st.queue, st.kernel, 1, nullptr,
                           &globalSize, &localSize, 0, nullptr, nullptr);

    std::vector<float> result(n);

    clEnqueueReadBuffer(st.queue, st.outBuf, CL_TRUE, 0, bytes,
                        result.data(), 0, nullptr, nullptr);

    return result;
}