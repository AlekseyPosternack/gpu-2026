#include "gelu_ocl.h"

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <vector>
#include <cmath>
#include <iostream>

constexpr const char* kernel_source = R"(
inline float gelu_func(float x) {
    float x3 = x * x * x;
    float z = 0.7978845608028654f * (x + 0.044715f * x3);

    return x / (1.0f + exp(-2.0f * z));
}

__kernel void gelu_vec(__global const float4* in, __global float4* out, int n4) {
    int i = get_global_id(0);
    if (i < n4) {
        float4 x = in[i];
        float4 res;
        res.x = gelu_func(x.x);
        res.y = gelu_func(x.y);
        res.z = gelu_func(x.z);
        res.w = gelu_func(x.w);
        out[i] = res;
    }
}

__kernel void gelu_scalar(__global const float* in, __global float* out, int n) {
    int i = get_global_id(0);
    if (i < n) {
        out[i] = gelu_func(in[i]);
    }
}
)";

struct OCLState {
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel_vec = nullptr;
    cl_kernel kernel_scalar = nullptr;
    cl_mem d_in = nullptr;
    cl_mem d_out = nullptr;
    size_t capacity_bytes = 0;
    int current_platform = -1;
    bool is_initialized = false;

    void cleanup() {
        if (kernel_vec) clReleaseKernel(kernel_vec);
        if (kernel_scalar) clReleaseKernel(kernel_scalar);
        if (program) clReleaseProgram(program);
        if (d_in) clReleaseMemObject(d_in);
        if (d_out) clReleaseMemObject(d_out);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
        
        context = nullptr;
        queue = nullptr;
        program = nullptr;
        kernel_vec = nullptr;
        kernel_scalar = nullptr;
        d_in = nullptr;
        d_out = nullptr;
        capacity_bytes = 0;
        current_platform = -1;
        is_initialized = false;
    }

    void init_opencl(int platform_idx) {
        if (is_initialized && current_platform == platform_idx) {
            return;
        }
        cleanup();

        cl_uint num_platforms;
        clGetPlatformIDs(0, nullptr, &num_platforms);
        if (num_platforms == 0) return;

        std::vector<cl_platform_id> platforms(num_platforms);
        clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

        int safe_plat = platform_idx;
        if (safe_plat < 0 || safe_plat >= (int)num_platforms) {
            safe_plat = 0;
        }

        cl_device_id device = nullptr;

        cl_int err = clGetDeviceIDs(platforms[safe_plat], CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

        if (err != CL_SUCCESS) {
            clGetDeviceIDs(platforms[safe_plat], CL_DEVICE_TYPE_ALL, 1, &device, nullptr);
        }

        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
        queue = clCreateCommandQueue(context, device, 0, nullptr);

        cl_program prog = clCreateProgramWithSource(context, 1, &kernel_source, nullptr, nullptr);

        clBuildProgram(prog, 1, &device, nullptr, nullptr, nullptr);

        kernel_vec = clCreateKernel(prog, "gelu_vec", nullptr);
        kernel_scalar = clCreateKernel(prog, "gelu_scalar", nullptr);
        
        clReleaseProgram(prog);
        current_platform = safe_plat;
        is_initialized = true;
    }

    void ensure_capacity(size_t bytes) {
        if (capacity_bytes >= bytes) {
            return;
        }
        if (d_in) clReleaseMemObject(d_in);
        if (d_out) clReleaseMemObject(d_out);
        
        d_in = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, nullptr, nullptr);
        d_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, nullptr);
        capacity_bytes = bytes;
    }
};

static OCLState state;

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    size_t n = input.size();
    if (n == 0) return {};

    size_t bytes = n * sizeof(float);

    state.init_opencl(platform);
    if (!state.is_initialized) return {};

    state.ensure_capacity(bytes);

    clEnqueueWriteBuffer(state.queue, state.d_in, CL_FALSE, 0, bytes, input.data(), 0, nullptr, nullptr);

    bool use_vec = (n % 4 == 0);
    cl_kernel kernel = use_vec ? state.kernel_vec : state.kernel_scalar;
    int count = use_vec ? static_cast<int>(n / 4) : static_cast<int>(n);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &state.d_in);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &state.d_out);
    clSetKernelArg(kernel, 2, sizeof(int), &count);

    size_t local_size = 256;
    size_t global_size = ((count + local_size - 1) / local_size) * local_size;

    clEnqueueNDRangeKernel(state.queue, kernel, 1, nullptr, &global_size, &local_size, 0, nullptr, nullptr);

    std::vector<float> result(n);

    clEnqueueReadBuffer(state.queue, state.d_out, CL_TRUE, 0, bytes, result.data(), 0, nullptr, nullptr);

    clFinish(state.queue);

    return result;
}