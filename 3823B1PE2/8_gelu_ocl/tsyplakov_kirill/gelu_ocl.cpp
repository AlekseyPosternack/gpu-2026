#include "gelu_ocl.h"
#include <CL/cl.h>
#include <cmath>
#include <cstddef>
#include <stdexcept>

static const char* kernel_source = R"(
__kernel void gelu_kernel(__global const float* input,
                          __global float* output,
                          const int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    float x = input[i];
    float arg = 0.7978845608f * (x + 0.044715f * x * x * x);
    float e = exp(-2.0f * arg);
    float tanh_val = (1.0f - e) / (1.0f + e);
    output[i] = 0.5f * x * (1.0f + tanh_val);
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    std::size_t n = input.size();
    std::vector<float> output(n);
    if (n == 0) return output;

    cl_int err;

    cl_platform_id platform_id;
    err = clGetPlatformIDs(1, &platform_id, nullptr);
    if (err != CL_SUCCESS) throw std::runtime_error("clGetPlatformIDs failed");

    cl_device_id device_id;
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, nullptr);
    if (err != CL_SUCCESS) {
        // Fallback на CPU (POCL)
        err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("No OpenCL device");
    }

    cl_context context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &err);
    cl_command_queue queue = clCreateCommandQueue(context, device_id, 0, &err);

    cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, nullptr, &err);
    err = clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) throw std::runtime_error("clBuildProgram failed");

    cl_kernel kernel = clCreateKernel(program, "gelu_kernel", &err);

    std::size_t bytes = n * sizeof(float);
    cl_mem d_in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, (void*)input.data(), &err);
    cl_mem d_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out);
    clSetKernelArg(kernel, 2, sizeof(int), &n);

    std::size_t global_size = n;
    std::size_t local_size = 256;
    global_size = ((n + local_size - 1) / local_size) * local_size;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, &local_size, 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0, bytes, output.data(), 0, nullptr, nullptr);

    clReleaseMemObject(d_in);
    clReleaseMemObject(d_out);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return output;
}