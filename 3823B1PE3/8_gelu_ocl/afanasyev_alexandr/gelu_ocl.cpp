#include "gelu_ocl.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace {
inline float GeluScalar(float x) noexcept {
    constexpr float kSqrt2OverPi = 0.7978845608f;
    constexpr float kGeluCubicCoeff = 0.044715f;

    const float x3 = x * x * x;
    const float tanh_arg = kSqrt2OverPi * (x + kGeluCubicCoeff * x3);
    return 0.5f * x * (1.0f + std::tanh(tanh_arg));
}

constexpr const char* kKernelSource = R"CLC(
__kernel void gelu_kernel(__global const float* input,
                         __global float* output) {
    const int gid = get_global_id(0);
    const float x = input[gid];
    const float sqrt2_over_pi = 0.7978845608f;
    const float gelu_cubic_coeff = 0.044715f;
    const float x3 = x * x * x;
    const float tanh_arg = sqrt2_over_pi * (x + gelu_cubic_coeff * x3);
    output[gid] = 0.5f * x * (1.0f + tanh(tanh_arg));
}
)CLC";

std::vector<cl_platform_id> GetPlatforms() {
    cl_uint platform_count = 0;
    if (clGetPlatformIDs(0, nullptr, &platform_count) != CL_SUCCESS || platform_count == 0) {
        return {};
    }

    std::vector<cl_platform_id> platforms(platform_count);
    if (clGetPlatformIDs(platform_count, platforms.data(), nullptr) != CL_SUCCESS) {
        return {};
    }
    return platforms;
}

bool GetGpuDevice(cl_platform_id platform, cl_device_id* device) {
    cl_uint device_count = 0;
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count) != CL_SUCCESS ||
        device_count == 0) {
        return false;
    }

    std::vector<cl_device_id> devices(device_count);
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, device_count, devices.data(), nullptr) != CL_SUCCESS) {
        return false;
    }

    *device = devices[0];
    return true;
}

std::vector<float> GeluCpuFallback(const std::vector<float>& input) {
    std::vector<float> result(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        result[i] = GeluScalar(input[i]);
    }
    return result;
}
}  // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (input.empty()) {
        return {};
    }

    const std::vector<cl_platform_id> platforms = GetPlatforms();
    if (platforms.empty()) {
        return GeluCpuFallback(input);
    }

    const int selected_platform = std::max(0, std::min(platform, static_cast<int>(platforms.size()) - 1));
    cl_device_id device = nullptr;
    if (!GetGpuDevice(platforms[selected_platform], &device)) {
        return GeluCpuFallback(input);
    }

    cl_int err = CL_SUCCESS;
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS || context == nullptr) {
        return GeluCpuFallback(input);
    }

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS || queue == nullptr) {
        clReleaseContext(context);
        return GeluCpuFallback(input);
    }

    const std::size_t size = input.size();
    cl_mem d_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   size * sizeof(float), const_cast<float*>(input.data()), &err);
    if (err != CL_SUCCESS || d_input == nullptr) {
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return GeluCpuFallback(input);
    }

    cl_mem d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                    size * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS || d_output == nullptr) {
        clReleaseMemObject(d_input);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return GeluCpuFallback(input);
    }

    const char* source = kKernelSource;
    cl_program program = clCreateProgramWithSource(context, 1, &source, nullptr, &err);
    if (err != CL_SUCCESS || program == nullptr) {
        clReleaseMemObject(d_output);
        clReleaseMemObject(d_input);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return GeluCpuFallback(input);
    }

    if (clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr) != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        if (log_size > 0) {
            std::string build_log(log_size, '\0');
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr);
        }

        clReleaseProgram(program);
        clReleaseMemObject(d_output);
        clReleaseMemObject(d_input);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return GeluCpuFallback(input);
    }

    cl_kernel kernel = clCreateKernel(program, "gelu_kernel", &err);
    if (err != CL_SUCCESS || kernel == nullptr) {
        clReleaseProgram(program);
        clReleaseMemObject(d_output);
        clReleaseMemObject(d_input);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return GeluCpuFallback(input);
    }

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);

    const std::size_t global_size = size;
    if (clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr) !=
        CL_SUCCESS) {
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseMemObject(d_output);
        clReleaseMemObject(d_input);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return GeluCpuFallback(input);
    }

    std::vector<float> result(size);
    if (clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, size * sizeof(float), result.data(), 0, nullptr,
                            nullptr) != CL_SUCCESS) {
        result = GeluCpuFallback(input);
    }

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(d_output);
    clReleaseMemObject(d_input);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return result;
}
