#include "gelu_ocl.h"

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <map>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

const char* kernel_source = R"CLC(
__kernel void gelu(__global float* data, ulong size) {
    size_t i = get_global_id(0);
    if (i < size) {
        float x = data[i];
        float exponent = 1.5957691216057308f * x * (1.0f + 0.044715f * x * x);
        data[i] = x / (1.0f + exp(-exponent));
    }
}
)CLC";

void checkOpenCL(cl_int error) {
    if (error != CL_SUCCESS) {
        throw std::runtime_error("OpenCL error: " + std::to_string(error));
    }
}

struct OpenCLState {
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_mem buffer = nullptr;
    std::size_t capacity = 0;

    ~OpenCLState() {
        if (buffer) clReleaseMemObject(buffer);
        if (kernel) clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }
};

OpenCLState& getState(int platform) {
    static thread_local std::map<int, std::unique_ptr<OpenCLState>> states;
    auto found = states.find(platform);
    if (found != states.end()) {
        return *found->second;
    }

    cl_uint platform_count = 0;
    checkOpenCL(clGetPlatformIDs(0, nullptr, &platform_count));
    if (platform < 0 || static_cast<cl_uint>(platform) >= platform_count) {
        throw std::invalid_argument("Invalid OpenCL platform index");
    }
    std::vector<cl_platform_id> platforms(platform_count);
    checkOpenCL(clGetPlatformIDs(platform_count, platforms.data(), nullptr));

    cl_device_id device = nullptr;
    checkOpenCL(clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 1, &device, nullptr));

    auto state = std::make_unique<OpenCLState>();
    cl_int error = CL_SUCCESS;
    state->context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error);
    checkOpenCL(error);
    state->queue = clCreateCommandQueue(state->context, device, 0, &error);
    checkOpenCL(error);
    state->program = clCreateProgramWithSource(state->context, 1, &kernel_source, nullptr, &error);
    checkOpenCL(error);
    error = clBuildProgram(state->program, 1, &device, nullptr, nullptr, nullptr);
    if (error != CL_SUCCESS) {
        std::size_t length = 0;
        checkOpenCL(clGetProgramBuildInfo(state->program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &length));
        std::string log(length, '\0');
        checkOpenCL(clGetProgramBuildInfo(state->program, device, CL_PROGRAM_BUILD_LOG, length, log.data(), nullptr));
        throw std::runtime_error("OpenCL kernel build failed: " + log);
    }
    state->kernel = clCreateKernel(state->program, "gelu", &error);
    checkOpenCL(error);

    auto inserted = states.emplace(platform, std::move(state));
    return *inserted.first->second;
}

}

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (platform < 0) {
        throw std::invalid_argument("Invalid OpenCL platform index");
    }
    if (input.empty()) {
        return {};
    }

    OpenCLState& state = getState(platform);
    const std::size_t bytes = input.size() * sizeof(float);
    if (bytes > state.capacity) {
        cl_int error = CL_SUCCESS;
        cl_mem buffer = clCreateBuffer(state.context, CL_MEM_READ_WRITE, bytes, nullptr, &error);
        checkOpenCL(error);
        if (state.buffer) clReleaseMemObject(state.buffer);
        state.buffer = buffer;
        state.capacity = bytes;
    }

    checkOpenCL(clEnqueueWriteBuffer(state.queue, state.buffer, CL_TRUE, 0, bytes, input.data(), 0, nullptr, nullptr));
    const cl_ulong size = static_cast<cl_ulong>(input.size());
    checkOpenCL(clSetKernelArg(state.kernel, 0, sizeof(cl_mem), &state.buffer));
    checkOpenCL(clSetKernelArg(state.kernel, 1, sizeof(size), &size));
    const std::size_t local_size = 256;
    const std::size_t global_size = (input.size() + local_size - 1) / local_size * local_size;
    checkOpenCL(clEnqueueNDRangeKernel(state.queue, state.kernel, 1, nullptr, &global_size, &local_size, 0, nullptr, nullptr));

    std::vector<float> result(input.size());
    checkOpenCL(clEnqueueReadBuffer(state.queue, state.buffer, CL_TRUE, 0, bytes, result.data(), 0, nullptr, nullptr));
    return result;
}
