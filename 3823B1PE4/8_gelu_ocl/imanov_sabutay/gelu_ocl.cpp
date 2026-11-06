#include "gelu_ocl.h"

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#include <CL/cl.h>

#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace {

const char* const kKernelSource = R"CLC(
__kernel void gelu(__global float* values) {
    const size_t i = get_global_id(0);
    const float x = values[i];
    const float argument = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    values[i] = x / (1.0f + exp(-2.0f * argument));
}
)CLC";

void CheckCL(cl_int status, const char* operation) {
    if (status != CL_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": OpenCL status " + std::to_string(status));
    }
}

struct State {
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_mem buffer = nullptr;
    std::size_t capacity = 0;

    explicit State(int platform_index) {
        try {
            cl_uint count = 0;
            CheckCL(clGetPlatformIDs(0, nullptr, &count), "clGetPlatformIDs");
            if (static_cast<unsigned int>(platform_index) >= count) {
                throw std::invalid_argument("OpenCL platform index is out of range");
            }
            std::vector<cl_platform_id> platforms(count);
            CheckCL(clGetPlatformIDs(count, platforms.data(), nullptr), "clGetPlatformIDs");
            cl_device_id device = nullptr;
            // Request only the first GPU from the selected platform.
            CheckCL(clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_GPU,
                                   1, &device, nullptr), "clGetDeviceIDs GPU 0");
            const cl_context_properties properties[] = {
                CL_CONTEXT_PLATFORM,
                reinterpret_cast<cl_context_properties>(platforms[platform_index]), 0
            };
            cl_int status = CL_SUCCESS;
            context = clCreateContext(properties, 1, &device, nullptr, nullptr, &status);
            CheckCL(status, "clCreateContext");
            queue = clCreateCommandQueue(context, device, 0, &status);
            CheckCL(status, "clCreateCommandQueue");
            const char* source = kKernelSource;
            program = clCreateProgramWithSource(context, 1, &source, nullptr, &status);
            CheckCL(status, "clCreateProgramWithSource");
            status = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
            if (status != CL_SUCCESS) {
                std::size_t log_size = 0;
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
                std::string log(log_size, '\0');
                if (log_size != 0) {
                    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                          log_size, &log[0], nullptr);
                }
                throw std::runtime_error("clBuildProgram: " + std::to_string(status) + "\n" + log);
            }
            kernel = clCreateKernel(program, "gelu", &status);
            CheckCL(status, "clCreateKernel");
        } catch (...) {
            Release();
            throw;
        }
    }

    State(const State&) = delete;
    State& operator=(const State&) = delete;
    ~State() { Release(); }

    void Release() noexcept {
        if (buffer != nullptr) clReleaseMemObject(buffer);
        if (kernel != nullptr) clReleaseKernel(kernel);
        if (program != nullptr) clReleaseProgram(program);
        if (queue != nullptr) clReleaseCommandQueue(queue);
        if (context != nullptr) clReleaseContext(context);
    }

    void Reserve(std::size_t bytes) {
        if (bytes <= capacity) return;
        cl_int status = CL_SUCCESS;
        cl_mem replacement = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, nullptr, &status);
        CheckCL(status, "clCreateBuffer");
        if (buffer != nullptr) clReleaseMemObject(buffer);
        buffer = replacement;
        capacity = bytes;
    }
};

} // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (platform < 0) throw std::invalid_argument("OpenCL platform index must be nonnegative");
    if (input.empty()) return {};
    if (input.size() > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::length_error("input is too large");
    }
    // Isolate mutable queues/kernels between calling threads; initialize once per platform.
    thread_local std::unordered_map<int, std::unique_ptr<State>> states;
    auto entry = states.find(platform);
    if (entry == states.end()) {
        auto state = std::make_unique<State>(platform);
        entry = states.emplace(platform, std::move(state)).first;
    }
    State& state = *entry->second;
    const std::size_t bytes = input.size() * sizeof(float);
    state.Reserve(bytes);
    // Blocking upload makes the input lifetime safe even if a later call fails.
    CheckCL(clEnqueueWriteBuffer(state.queue, state.buffer, CL_TRUE, 0, bytes,
                                 input.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer");
    CheckCL(clSetKernelArg(state.kernel, 0, sizeof(state.buffer), &state.buffer), "clSetKernelArg");
    const std::size_t global_size = input.size();
    CheckCL(clEnqueueNDRangeKernel(state.queue, state.kernel, 1, nullptr, &global_size,
                                   nullptr, 0, nullptr, nullptr), "clEnqueueNDRangeKernel");
    CheckCL(clFlush(state.queue), "clFlush");

    // GPU work can run while the CPU allocates the result vector.
    std::vector<float> output(input.size());
    CheckCL(clEnqueueReadBuffer(state.queue, state.buffer, CL_TRUE, 0, bytes,
                                output.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer");
    return output;
}
