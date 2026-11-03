#include "gelu_ocl.h"

#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace {

const char* kSource = R"CLC(
__kernel void gelu(__global const float* in, __global float* out, const ulong n) {
    const size_t i = get_global_id(0);
    if (i < n) {
        const float a = -1.5957691216f;
        const float b = a * 0.044715f;
        const float x = in[i];
        out[i] = x / (1.0f + native_exp(x * (a + b * x * x)));
    }
}
)CLC";

void Check(cl_int err, const char* what) {
    if (err != CL_SUCCESS)
        throw std::runtime_error(std::string("OpenCL error ") +
                                 std::to_string(err) + " in " + what);
}

struct State {
    int platform = -1;
    cl_context ctx = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_mem d_in = nullptr;
    cl_mem d_out = nullptr;
    size_t capacity = 0;
};

State g;

void Init(int platform_idx) {
    if (g.platform == platform_idx) return;
    if (g.platform != -1)
        throw std::runtime_error("GeluOCL: platform changed between calls");

    cl_uint num_platforms = 0;
    Check(clGetPlatformIDs(0, nullptr, &num_platforms), "clGetPlatformIDs");
    if (platform_idx < 0 || static_cast<cl_uint>(platform_idx) >= num_platforms)
        throw std::runtime_error("GeluOCL: bad platform index");
    std::vector<cl_platform_id> platforms(num_platforms);
    Check(clGetPlatformIDs(num_platforms, platforms.data(), nullptr),
          "clGetPlatformIDs");

    cl_device_id device;
    Check(clGetDeviceIDs(platforms[platform_idx], CL_DEVICE_TYPE_GPU, 1, &device,
                         nullptr),
          "clGetDeviceIDs");

    cl_int err;
    g.ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    Check(err, "clCreateContext");
    g.queue = clCreateCommandQueue(g.ctx, device, 0, &err);
    Check(err, "clCreateCommandQueue");
    g.program = clCreateProgramWithSource(g.ctx, 1, &kSource, nullptr, &err);
    Check(err, "clCreateProgramWithSource");
    Check(clBuildProgram(g.program, 1, &device, "", nullptr, nullptr),
          "clBuildProgram");
    g.kernel = clCreateKernel(g.program, "gelu", &err);
    Check(err, "clCreateKernel");
    g.platform = platform_idx;
}

void Reserve(size_t n) {
    if (n <= g.capacity) return;
    cl_int err;
    if (g.d_in) clReleaseMemObject(g.d_in);
    if (g.d_out) clReleaseMemObject(g.d_out);
    g.d_in = clCreateBuffer(g.ctx, CL_MEM_READ_ONLY, n * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(in)");
    g.d_out = clCreateBuffer(g.ctx, CL_MEM_WRITE_ONLY, n * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(out)");
    g.capacity = n;
}

}  // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    const size_t n = input.size();
    if (n == 0) return {};

    Init(platform);
    Reserve(n);

    const size_t bytes = n * sizeof(float);
    Check(clEnqueueWriteBuffer(g.queue, g.d_in, CL_TRUE, 0, bytes, input.data(),
                               0, nullptr, nullptr),
          "write");

    cl_ulong n_arg = n;
    Check(clSetKernelArg(g.kernel, 0, sizeof(cl_mem), &g.d_in), "arg0");
    Check(clSetKernelArg(g.kernel, 1, sizeof(cl_mem), &g.d_out), "arg1");
    Check(clSetKernelArg(g.kernel, 2, sizeof(cl_ulong), &n_arg), "arg2");

    const size_t local = 256;
    const size_t global = (n + local - 1) / local * local;
    Check(clEnqueueNDRangeKernel(g.queue, g.kernel, 1, nullptr, &global, &local,
                                 0, nullptr, nullptr),
          "kernel");

    std::vector<float> output(n);

    Check(clEnqueueReadBuffer(g.queue, g.d_out, CL_TRUE, 0, bytes, output.data(),
                              0, nullptr, nullptr),
          "read");
    return output;
}
