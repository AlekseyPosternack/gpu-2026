#include "gelu_ocl.h"
#include <CL/cl.h>
#include <iostream>
#include <cmath>

const char* ocl_kernel_source = R"(
__kernel void gelu_kernel(__global const float* input,
                          __global float* output,
                          const int size) {
    int i = get_global_id(0);
    if (i < size) {
        float x = input[i];
        float z = 0.7978845608028654f * (x + 0.044715f * x * x * x);
        output[i] = x / (1.0f + exp(-2.0f * z));
    }
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform_id) {
    const int size = static_cast<int>(input.size());
    std::vector<float> output(size, 0.0f);

    if (size == 0) return output;

    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr); 

    cl_platform_id platforms[10];
    cl_uint num_platforms;
    clGetPlatformIDs(10, platforms, &num_platforms);
    
    if (platform_id < num_platforms) {
        platform = platforms[platform_id];
    }

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, nullptr);

    cl_program program = clCreateProgramWithSource(context, 1, &ocl_kernel_source, nullptr, nullptr);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

    cl_kernel kernel = clCreateKernel(program, "gelu_kernel", nullptr);

    cl_mem d_input = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * size, nullptr, nullptr);
    cl_mem d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * size, nullptr, nullptr);

    clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0, sizeof(float) * size, input.data(), 0, nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
    clSetKernelArg(kernel, 2, sizeof(int), &size);

    size_t global_work_size = size;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);

    clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, sizeof(float) * size, output.data(), 0, nullptr, nullptr);

    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return output;
}