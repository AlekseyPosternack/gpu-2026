#include "gelu_ocl.h"

#include <CL/cl.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

static const char* GELU_KERNEL_SRC = R"CLC(
__kernel void gelu_kernel(__global const float* in,
                          __global float* out,
                          const int n)
{
    int i = get_global_id(0);
    if (i >= n) return;

    float x = in[i];
    float x3 = x * x * x;
    float arg = 1.5957691216057308f * (x + 0.044715f * x3);
    out[i] = x / (1.0f + exp(-arg));
}
)CLC";

namespace {

struct OclState {
    bool     initialized = false;
    cl_context       ctx    = nullptr;
    cl_command_queue queue  = nullptr;
    cl_program       prog   = nullptr;
    cl_kernel        kernel = nullptr;

    cl_mem d_in  = nullptr;
    cl_mem d_out = nullptr;
    size_t capacity = 0;
};

OclState g_state;

void check(cl_int err, const char* what) {
    if (err != CL_SUCCESS) {
        throw std::runtime_error(std::string(what) + " failed, code = "
                                 + std::to_string(err));
    }
}

void ensure_initialized(int platform)
{
    if (g_state.initialized) return;

    cl_uint num_platforms = 0;
    check(clGetPlatformIDs(0, nullptr, &num_platforms),
          "clGetPlatformIDs(count)");
    if (num_platforms == 0)
        throw std::runtime_error("No OpenCL platforms found");
    if (platform < 0 || (cl_uint)platform >= num_platforms)
        throw std::runtime_error("Invalid platform index");

    std::vector<cl_platform_id> ids(num_platforms);
    check(clGetPlatformIDs(num_platforms, ids.data(), nullptr),
          "clGetPlatformIDs(list)");
    cl_platform_id chosen = ids[platform];

    cl_device_id device = nullptr;
    check(clGetDeviceIDs(chosen, CL_DEVICE_TYPE_GPU, 1, &device, nullptr),
          "clGetDeviceIDs(GPU)");

    g_state.ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    check(g_state.ctx ? CL_SUCCESS : CL_INVALID_CONTEXT, "clCreateContext");

    g_state.queue = clCreateCommandQueue(g_state.ctx, device, 0, nullptr);
    check(g_state.queue ? CL_SUCCESS : CL_INVALID_COMMAND_QUEUE,
          "clCreateCommandQueue");

    g_state.prog = clCreateProgramWithSource(
        g_state.ctx, 1, &GELU_KERNEL_SRC, nullptr, nullptr);
    check(g_state.prog ? CL_SUCCESS : CL_INVALID_PROGRAM,
          "clCreateProgramWithSource");

    check(clBuildProgram(g_state.prog, 1, &device, "-cl-fast-relaxed-math",
                         nullptr, nullptr),
          "clBuildProgram");

    g_state.kernel = clCreateKernel(g_state.prog, "gelu_kernel", nullptr);
    check(g_state.kernel ? CL_SUCCESS : CL_INVALID_KERNEL, "clCreateKernel");

    g_state.initialized = true;
}

void ensure_buffers(size_t n)
{
    if (n <= g_state.capacity) return;
    if (g_state.d_in)  clReleaseMemObject(g_state.d_in);
    if (g_state.d_out) clReleaseMemObject(g_state.d_out);
    const size_t bytes = n * sizeof(float);
    g_state.d_in  = clCreateBuffer(g_state.ctx, CL_MEM_READ_ONLY,
                                   bytes, nullptr, nullptr);
    g_state.d_out = clCreateBuffer(g_state.ctx, CL_MEM_WRITE_ONLY,
                                   bytes, nullptr, nullptr);
    g_state.capacity = n;
}

} // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform)
{
    const size_t n = input.size();
    std::vector<float> output(n);
    if (n == 0) return output;

    ensure_initialized(platform);
    ensure_buffers(n);

    const size_t bytes = n * sizeof(float);

    check(clEnqueueWriteBuffer(g_state.queue, g_state.d_in, CL_TRUE,
                               0, bytes, input.data(),
                               0, nullptr, nullptr),
          "clEnqueueWriteBuffer");

    cl_int n_int = (cl_int)n;
    check(clSetKernelArg(g_state.kernel, 0, sizeof(cl_mem), &g_state.d_in),
          "clSetKernelArg(in)");
    check(clSetKernelArg(g_state.kernel, 1, sizeof(cl_mem), &g_state.d_out),
          "clSetKernelArg(out)");
    check(clSetKernelArg(g_state.kernel, 2, sizeof(cl_int), &n_int),
          "clSetKernelArg(n)");

    const size_t local_size = 256;
    const size_t global_size = ((n + local_size - 1) / local_size) * local_size;
    check(clEnqueueNDRangeKernel(g_state.queue, g_state.kernel,
                                 1, nullptr, &global_size, &local_size,
                                 0, nullptr, nullptr),
          "clEnqueueNDRangeKernel");

    check(clEnqueueReadBuffer(g_state.queue, g_state.d_out, CL_TRUE,
                              0, bytes, output.data(),
                              0, nullptr, nullptr),
          "clEnqueueReadBuffer");

    return output;
}