#include "gelu_ocl.h"

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

namespace {

constexpr char source[] = R"CLC(
__kernel void gelu(__global const float* input,
                   __global float* output, ulong n) {
    size_t i = get_global_id(0);

    if (i < n) {
        float x = input[i];
        float z = 0.7978845608f * (x + 0.044715f * x * x * x);
        output[i] = x / (1.0f + exp(-2.0f * z));
    }
}
)CLC";

struct OpenCLData {
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_mem input = nullptr;
    cl_mem output = nullptr;

    int platform = -1;
    std::size_t capacity = 0;

    void Clear() {
        if (input) clReleaseMemObject(input);
        if (output) clReleaseMemObject(output);
        if (kernel) clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);

        input = output = nullptr;
        kernel = nullptr;
        program = nullptr;
        queue = nullptr;
        context = nullptr;
        platform = -1;
        capacity = 0;
    }

    ~OpenCLData() {
        Clear();
    }
};

}  // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (input.empty()) return {};

    static thread_local OpenCLData data;

    if (!data.kernel || data.platform != platform) {
        data.Clear();

        cl_uint count = 0;
        clGetPlatformIDs(0, nullptr, &count);

        std::vector<cl_platform_id> platforms(count);
        clGetPlatformIDs(count, platforms.data(), nullptr);

        cl_device_id device = nullptr;
        clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU,
                       1, &device, nullptr);

        cl_context_properties properties[] = {
            CL_CONTEXT_PLATFORM,
            reinterpret_cast<cl_context_properties>(platforms[platform]),
            0
        };

        data.context = clCreateContext(
            properties, 1, &device, nullptr, nullptr, nullptr);

        data.queue = clCreateCommandQueue(
            data.context, device, 0, nullptr);

        const char* code = source;
        data.program = clCreateProgramWithSource(
            data.context, 1, &code, nullptr, nullptr);

        clBuildProgram(data.program, 1, &device, nullptr, nullptr, nullptr);
        data.kernel = clCreateKernel(data.program, "gelu", nullptr);

        data.platform = platform;
    }

    const std::size_t bytes = input.size() * sizeof(float);

    if (bytes > data.capacity) {
        if (data.input) clReleaseMemObject(data.input);
        if (data.output) clReleaseMemObject(data.output);

        data.input = clCreateBuffer(
            data.context, CL_MEM_READ_ONLY, bytes, nullptr, nullptr);

        data.output = clCreateBuffer(
            data.context, CL_MEM_WRITE_ONLY, bytes, nullptr, nullptr);

        data.capacity = bytes;
    }

    clEnqueueWriteBuffer(
        data.queue, data.input, CL_TRUE, 0, bytes,
        input.data(), 0, nullptr, nullptr);

    const cl_ulong n = static_cast<cl_ulong>(input.size());

    clSetKernelArg(data.kernel, 0, sizeof(cl_mem), &data.input);
    clSetKernelArg(data.kernel, 1, sizeof(cl_mem), &data.output);
    clSetKernelArg(data.kernel, 2, sizeof(n), &n);

    const std::size_t local = 256;
    const std::size_t global = ((input.size() + local - 1) / local) * local;

    clEnqueueNDRangeKernel(
        data.queue, data.kernel, 1, nullptr,
        &global, &local, 0, nullptr, nullptr);

    clFlush(data.queue);

    std::vector<float> output(input.size());

    clEnqueueReadBuffer(
        data.queue, data.output, CL_TRUE, 0, bytes,
        output.data(), 0, nullptr, nullptr);

    return output;
}
