#include "gelu_ocl.h"
#include <CL/cl.h>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
const char* kSource = R"CLC(
__kernel void gelu(__global const float* input, __global float* output, uint size) {
    const uint i = get_global_id(0); if (i >= size) return;
    const float x = input[i], x3 = x * x * x;
    output[i] = 0.5f * x * (1.0f + tanh(0.7978845608028654f * (x + 0.044715f * x3)));
}
)CLC";
void Check(cl_int e, const char* what) { if (e != CL_SUCCESS) throw std::runtime_error(std::string(what) + ": " + std::to_string(e)); }
void Release(cl_mem& object) { if (object) { clReleaseMemObject(object); object = nullptr; } }
}

std::vector<float> GeluOCL(const std::vector<float>& input, int platform_index) {
    std::vector<float> output(input.size()); if (input.empty()) return output;
    cl_uint count = 0; Check(clGetPlatformIDs(0, nullptr, &count), "clGetPlatformIDs");
    if (platform_index < 0 || platform_index >= static_cast<int>(count)) throw std::invalid_argument("invalid OpenCL platform");
    std::vector<cl_platform_id> platforms(count); Check(clGetPlatformIDs(count, platforms.data(), nullptr), "clGetPlatformIDs");
    cl_uint device_count = 0; Check(clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count), "clGetDeviceIDs");
    if (!device_count) throw std::runtime_error("no GPU device on selected OpenCL platform");
    std::vector<cl_device_id> devices(device_count); Check(clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_GPU, device_count, devices.data(), nullptr), "clGetDeviceIDs");
    cl_int status = CL_SUCCESS; cl_device_id device = devices[0];
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status); Check(status, "clCreateContext");
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &status); Check(status, "clCreateCommandQueue");
    const char* source = kSource; const std::size_t source_length = std::strlen(kSource);
    cl_program program = clCreateProgramWithSource(context, 1, &source, &source_length, &status); Check(status, "clCreateProgramWithSource");
    try {
        status = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        if (status != CL_SUCCESS) {
            std::size_t log_size = 0; clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::string log(log_size, '\0'); clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            throw std::runtime_error("OpenCL build failed: " + log);
        }
        cl_kernel kernel = clCreateKernel(program, "gelu", &status); Check(status, "clCreateKernel");
        cl_mem device_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input.size() * sizeof(float), const_cast<float*>(input.data()), &status); Check(status, "input buffer");
        cl_mem device_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output.size() * sizeof(float), nullptr, &status); Check(status, "output buffer");
        try {
            const cl_uint size = static_cast<cl_uint>(input.size());
            Check(clSetKernelArg(kernel, 0, sizeof(device_input), &device_input), "kernel input"); Check(clSetKernelArg(kernel, 1, sizeof(device_output), &device_output), "kernel output"); Check(clSetKernelArg(kernel, 2, sizeof(size), &size), "kernel size");
            const std::size_t local = 256, global = ((input.size() + local - 1) / local) * local;
            Check(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr), "clEnqueueNDRangeKernel");
            Check(clEnqueueReadBuffer(queue, device_output, CL_TRUE, 0, output.size() * sizeof(float), output.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer");
        } catch (...) { Release(device_output); Release(device_input); clReleaseKernel(kernel); throw; }
        Release(device_output); Release(device_input); clReleaseKernel(kernel);
    } catch (...) { clReleaseProgram(program); clReleaseCommandQueue(queue); clReleaseContext(context); throw; }
    clReleaseProgram(program); clReleaseCommandQueue(queue); clReleaseContext(context); return output;
}
