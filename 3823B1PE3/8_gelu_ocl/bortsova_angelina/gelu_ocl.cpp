#include "gelu_ocl.h"
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <memory>

namespace {

constexpr size_t WORK_GROUP_SIZE = 256;
constexpr char GELU_SOURCE[] = R"CLC(
__kernel void gelu_vector(__global const float* input, __global float* output, ulong element_count) {
    size_t first = get_global_id(0) * 4;
    if (first >= element_count) {
        return;
    }

    float4 values = (float4)(0.0f);
    if (first + 3 < element_count) {
        values = vload4(0, input + first);
    } else {
        for (int lane = 0; lane < 4 && first + lane < element_count; ++lane) {
            values[lane] = input[first + lane];
        }
    }

    float4 argument = 0.7978845608f * (values + 0.044715f * values * values * values);
    float4 result = 0.5f * values * (1.0f + tanh(argument));
    if (first + 3 < element_count) {
        vstore4(result, 0, output + first);
    } else {
        for (int lane = 0; lane < 4 && first + lane < element_count; ++lane) {
            output[first + lane] = result[lane];
        }
    }
}
)CLC";

struct GeluWorkspace {
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem input_buffer = nullptr;
    cl_mem output_buffer = nullptr;
    size_t capacity = 0;

    explicit GeluWorkspace(int platform_index) {
        cl_uint platform_count = 0;
        clGetPlatformIDs(0, nullptr, &platform_count);
        std::vector<cl_platform_id> platforms(platform_count);
        clGetPlatformIDs(platform_count, platforms.data(), nullptr);
        cl_device_id device;
        clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
        queue = clCreateCommandQueue(context, device, 0, nullptr);
        const char* source = GELU_SOURCE;
        program = clCreateProgramWithSource(context, 1, &source, nullptr, nullptr);
        clBuildProgram(program, 1, &device, "-cl-std=CL1.2", nullptr, nullptr);
        kernel = clCreateKernel(program, "gelu_vector", nullptr);
    }

    ~GeluWorkspace() {
        if (input_buffer) {
            clReleaseMemObject(input_buffer);
            clReleaseMemObject(output_buffer);
        }
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
    }

    void Reserve(size_t byte_count) {
        if (byte_count > capacity) {
            if (input_buffer) {
                clReleaseMemObject(input_buffer);
                clReleaseMemObject(output_buffer);
            }
            input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, byte_count, nullptr, nullptr);
            output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, byte_count, nullptr, nullptr);
            capacity = byte_count;
        }
    }
};

}

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (input.empty()) {
        return {};
    }

    static std::vector<std::unique_ptr<GeluWorkspace>> workspaces;
    if (workspaces.size() <= static_cast<size_t>(platform)) {
        workspaces.resize(platform + 1);
    }
    if (!workspaces[platform]) {
        workspaces[platform] = std::make_unique<GeluWorkspace>(platform);
    }
    GeluWorkspace& workspace = *workspaces[platform];
    size_t byte_count = input.size() * sizeof(float);
    workspace.Reserve(byte_count);
    cl_ulong element_count = input.size();
    clSetKernelArg(workspace.kernel, 0, sizeof(cl_mem), &workspace.input_buffer);
    clSetKernelArg(workspace.kernel, 1, sizeof(cl_mem), &workspace.output_buffer);
    clSetKernelArg(workspace.kernel, 2, sizeof(element_count), &element_count);
    size_t vector_count = (input.size() + 3) / 4;
    size_t global_size = (vector_count + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE * WORK_GROUP_SIZE;
    clEnqueueWriteBuffer(workspace.queue, workspace.input_buffer, CL_FALSE, 0, byte_count, input.data(), 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(workspace.queue, workspace.kernel, 1, nullptr, &global_size, &WORK_GROUP_SIZE, 0, nullptr, nullptr);
    clFlush(workspace.queue);

    std::vector<float> result(input.size());
    clEnqueueReadBuffer(workspace.queue, workspace.output_buffer, CL_TRUE, 0, byte_count, result.data(), 0, nullptr, nullptr);
    return result;
}
