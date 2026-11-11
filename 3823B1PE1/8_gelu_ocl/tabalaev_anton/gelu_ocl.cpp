#include "gelu_ocl.h"

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <vector>

namespace {

static const char* kernelSource = R"(
__kernel void gelu_kernel(__global const float* in,
                          __global float* out,
                          const int n) {
    int i = get_global_id(0);
    if (i < n) {
        float x = in[i];
        out[i] = x / (1.0f + exp(x * (-1.59576912f + -0.07135482f * x * x)));
    }
}
)";

static int g_platform_idx = -1;
static cl_context g_context = nullptr;
static cl_command_queue g_queue = nullptr;
static cl_program g_program = nullptr;
static cl_kernel g_kernel = nullptr;

static cl_mem g_d_in = nullptr;
static cl_mem g_d_out = nullptr;
static size_t g_allocated_bytes = 0;

void ensure_opencl_resources(int platform_idx, size_t required_bytes) {
    if (!g_context || g_platform_idx != platform_idx) {
        if (g_d_in) { clReleaseMemObject(g_d_in); g_d_in = nullptr; }
        if (g_d_out) { clReleaseMemObject(g_d_out); g_d_out = nullptr; }
        if (g_kernel) { clReleaseKernel(g_kernel); g_kernel = nullptr; }
        if (g_program) { clReleaseProgram(g_program); g_program = nullptr; }
        if (g_queue) { clReleaseCommandQueue(g_queue); g_queue = nullptr; }
        if (g_context) { clReleaseContext(g_context); g_context = nullptr; }
        g_allocated_bytes = 0;

        cl_uint numPlatforms = 0;
        clGetPlatformIDs(0, NULL, &numPlatforms);
        std::vector<cl_platform_id> platforms(numPlatforms);
        clGetPlatformIDs(numPlatforms, platforms.data(), NULL);

        cl_device_id device = NULL;
        clGetDeviceIDs(platforms[platform_idx], CL_DEVICE_TYPE_GPU, 1, &device, NULL);

        g_context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
        g_queue = clCreateCommandQueueWithProperties(g_context, device, NULL, NULL);

        g_program = clCreateProgramWithSource(g_context, 1, &kernelSource, NULL, NULL);
        clBuildProgram(g_program, 1, &device, NULL, NULL, NULL);
        g_kernel = clCreateKernel(g_program, "gelu_kernel", NULL);

        g_platform_idx = platform_idx;
    }

    if (required_bytes > g_allocated_bytes) {
        if (g_d_in) clReleaseMemObject(g_d_in);
        if (g_d_out) clReleaseMemObject(g_d_out);

        g_d_in = clCreateBuffer(g_context, CL_MEM_READ_ONLY, required_bytes, NULL, NULL);
        g_d_out = clCreateBuffer(g_context, CL_MEM_WRITE_ONLY, required_bytes, NULL, NULL);
        g_allocated_bytes = required_bytes;
    }
}

} // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    const size_t n = input.size();
    if (n == 0) {
        return {};
    }

    const size_t bytes = n * sizeof(float);
    ensure_opencl_resources(platform, bytes);

    clEnqueueWriteBuffer(g_queue, g_d_in, CL_FALSE, 0, bytes, input.data(), 0, NULL, NULL);

    const int count = static_cast<int>(n);
    clSetKernelArg(g_kernel, 0, sizeof(cl_mem), &g_d_in);
    clSetKernelArg(g_kernel, 1, sizeof(cl_mem), &g_d_out);
    clSetKernelArg(g_kernel, 2, sizeof(int), &count);

    const size_t localSize = 256;
    const size_t globalSize = ((n + localSize - 1) / localSize) * localSize;
    clEnqueueNDRangeKernel(g_queue, g_kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

    std::vector<float> output(n);

    clEnqueueReadBuffer(g_queue, g_d_out, CL_TRUE, 0, bytes, output.data(), 0, NULL, NULL);

    return output;
}
