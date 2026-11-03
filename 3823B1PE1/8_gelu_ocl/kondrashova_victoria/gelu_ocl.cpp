#include "gelu_ocl.h"

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

const char* kGeluKernelSource = R"(
__kernel void gelu(__global const float* input,
                   __global float* output,
                   const int size) {
    int idx = get_global_id(0);

    if (idx < size) {
        float x = input[idx];

        float t = 0.7978845608f * (x + 0.044715f * x * x * x);

        float e = exp(2.0f * t);
        float tanh_t = (e - 1.0f) / (e + 1.0f);

        output[idx] = 0.5f * x * (1.0f + tanh_t);
    }
}
)";

struct OpenCLContext {
    cl_device_id device_id = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
};

OpenCLContext CreateOpenCLContext(int platform_index) {
    OpenCLContext ocl;

    cl_uint platform_count = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &platform_count);

    if (err != CL_SUCCESS || platform_count == 0) {
        throw std::runtime_error("No OpenCL platforms found");
    }

    std::vector<cl_platform_id> platforms(platform_count);
    err = clGetPlatformIDs(platform_count, platforms.data(), nullptr);

    if (platform_index < 0 ||
        static_cast<size_t>(platform_index) >= platforms.size()) {
        throw std::runtime_error("Invalid OpenCL platform index");
    }

    cl_platform_id platform_id = platforms[platform_index];

    cl_context_properties properties[] = {
        CL_CONTEXT_PLATFORM,
        reinterpret_cast<cl_context_properties>(platform_id),
        0
    };

    ocl.context = clCreateContextFromType(properties, CL_DEVICE_TYPE_GPU,
                                           nullptr, nullptr, &err);

    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create OpenCL GPU context");
    }

    size_t devices_bytes = 0;
    err = clGetContextInfo(ocl.context, CL_CONTEXT_DEVICES, 0, nullptr,
                           &devices_bytes);

    std::vector<cl_device_id> devices(devices_bytes / sizeof(cl_device_id));
    err = clGetContextInfo(ocl.context, CL_CONTEXT_DEVICES, devices_bytes,
                           devices.data(), nullptr);

    if (devices.empty()) {
        throw std::runtime_error("No OpenCL GPU devices found");
    }

    ocl.device_id = devices[0];

    ocl.queue = clCreateCommandQueue(ocl.context, ocl.device_id, 0, &err);

    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create OpenCL command queue");
    }

    ocl.program = clCreateProgramWithSource(
        ocl.context, 1, &kGeluKernelSource, nullptr, &err);

    err = clBuildProgram(ocl.program, 1, &ocl.device_id, nullptr,
                         nullptr, nullptr);

    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to build OpenCL program");
    }

    ocl.kernel = clCreateKernel(ocl.program, "gelu", &err);

    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create OpenCL kernel");
    }

    return ocl;
}

OpenCLContext& GetOpenCLContext(int platform_index) {
    static OpenCLContext context = CreateOpenCLContext(platform_index);
    return context;
}

} // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    OpenCLContext& ocl = GetOpenCLContext(platform);

    const int size = static_cast<int>(input.size());
    std::vector<float> output(size);

    if (size == 0) {
        return output;
    }

    const size_t bytes = static_cast<size_t>(size) * sizeof(float);

    cl_int err;

    cl_mem d_input = clCreateBuffer(
        ocl.context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        bytes,
        const_cast<float*>(input.data()),
        &err);

    cl_mem d_output = clCreateBuffer(
        ocl.context,
        CL_MEM_WRITE_ONLY,
        bytes,
        nullptr,
        &err);

    err = clSetKernelArg(ocl.kernel, 0, sizeof(cl_mem), &d_input);
    err |= clSetKernelArg(ocl.kernel, 1, sizeof(cl_mem), &d_output);
    err |= clSetKernelArg(ocl.kernel, 2, sizeof(int), &size);

    const size_t global_size = static_cast<size_t>(size);

    err = clEnqueueNDRangeKernel(
        ocl.queue, ocl.kernel, 1, nullptr,
        &global_size, nullptr, 0, nullptr, nullptr);

    err = clEnqueueReadBuffer(
        ocl.queue, d_output, CL_TRUE, 0, bytes,
        output.data(), 0, nullptr, nullptr);

    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);

    return output;
}
