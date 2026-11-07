#include "gelu_ocl.h"

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const char* kSource = R"(
__kernel void gelu(__global const float* input, __global float* output, const int n) {
    int i = get_global_id(0);
    if (i < n) {
        float x = input[i];
        float x3 = x * x * x;
        float arg = 0.7978845608f * (x + 0.044715f * x3);
        output[i] = x / (1.0f + exp(-2.0f * arg));
    }
}
)";

void Check(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {
        throw std::runtime_error(std::string(msg) + " (error code: " + std::to_string(err) + ")");
    }
}

struct State {
    int platform = -1;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_kernel kernel = nullptr;
    cl_mem in = nullptr;
    cl_mem out = nullptr;
    size_t capacity = 0;

    void Release() {
        if (in) clReleaseMemObject(in);
        if (out) clReleaseMemObject(out);
        if (kernel) clReleaseKernel(kernel);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
        in = nullptr;
        out = nullptr;
        kernel = nullptr;
        queue = nullptr;
        context = nullptr;
        capacity = 0;
        platform = -1;
    }

    ~State() {
        Release();
    }
};

State g_state;
std::mutex g_mutex;

void Init(int platform) {
    g_state.Release();

    cl_uint num_platforms = 0;
    Check(clGetPlatformIDs(0, nullptr, &num_platforms), "clGetPlatformIDs(count)");
    if (platform < 0 || platform >= static_cast<int>(num_platforms)) {
        throw std::runtime_error("Invalid platform index");
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    Check(clGetPlatformIDs(num_platforms, platforms.data(), nullptr), "clGetPlatformIDs(list)");
    cl_platform_id platform_id = platforms[platform];

    cl_device_id device;
    Check(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device, nullptr), "No GPU device found");

    cl_int err;
    g_state.context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    Check(err, "clCreateContext");

    g_state.queue = clCreateCommandQueue(g_state.context, device, 0, &err);
    Check(err, "clCreateCommandQueue");

    cl_program program = clCreateProgramWithSource(g_state.context, 1, &kSource, nullptr, &err);
    Check(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        clReleaseProgram(program);
        throw std::runtime_error("OpenCL build failed:\n" + log);
    }

    g_state.kernel = clCreateKernel(program, "gelu", &err);
    clReleaseProgram(program);
    Check(err, "clCreateKernel");

    g_state.platform = platform;
}

void EnsureBuffers(size_t bytes) {
    if (g_state.capacity >= bytes) return;

    if (g_state.in) clReleaseMemObject(g_state.in);
    if (g_state.out) clReleaseMemObject(g_state.out);

    cl_int err;
    g_state.in = clCreateBuffer(g_state.context, CL_MEM_READ_ONLY, bytes, nullptr, &err);
    Check(err, "input buffer");
    g_state.out = clCreateBuffer(g_state.context, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
    Check(err, "output buffer");
    g_state.capacity = bytes;
}

}

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (input.empty()) return {};

    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_state.platform != platform) {
        Init(platform);
    }

    const size_t n = input.size();
    const size_t bytes = n * sizeof(float);
    EnsureBuffers(bytes);

    Check(clEnqueueWriteBuffer(g_state.queue, g_state.in, CL_TRUE, 0, bytes, input.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer");

    int n_int = static_cast<int>(n);
    Check(clSetKernelArg(g_state.kernel, 0, sizeof(cl_mem), &g_state.in), "clSetKernelArg(0)");
    Check(clSetKernelArg(g_state.kernel, 1, sizeof(cl_mem), &g_state.out), "clSetKernelArg(1)");
    Check(clSetKernelArg(g_state.kernel, 2, sizeof(int), &n_int), "clSetKernelArg(2)");

    size_t global_size = n;
    Check(clEnqueueNDRangeKernel(g_state.queue, g_state.kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr), "clEnqueueNDRangeKernel");

    std::vector<float> output(n);

    Check(clEnqueueReadBuffer(g_state.queue, g_state.out, CL_TRUE, 0, bytes, output.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer");

    return output;
}
