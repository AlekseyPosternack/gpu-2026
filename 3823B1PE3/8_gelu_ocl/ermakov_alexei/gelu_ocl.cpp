#include "gelu_ocl.h"

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#include <CL/cl.h>

#include <future>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

const char source[] = R"CLC(
float gelu_value(float x) {
    float z = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    return (0.5f * x) * (1.0f + tanh(z));
}

__kernel void gelu(__global float* data, ulong count) {
    size_t offset = get_global_id(0) * 4;
    if (offset >= count) {
        return;
    }
    if (count - offset >= 4) {
        float4 x = vload4(0, data + offset);
        float4 z = 0.7978845608028654f * (x + 0.044715f * x * x * x);
        float4 result = (0.5f * x) * (1.0f + tanh(z));
        vstore4(result, 0, data + offset);
    } else {
        for (size_t i = offset; i < count; ++i) {
            data[i] = gelu_value(data[i]);
        }
    }
}
)CLC";

void Check(cl_int status, const char* operation) {
    if (status != CL_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": " + std::to_string(status));
    }
}

struct State {
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_mem buffer = nullptr;
    size_t capacity = 0;

    ~State() {
        if (queue) {
            clFinish(queue);
        }
        if (buffer) {
            clReleaseMemObject(buffer);
        }
        if (kernel) {
            clReleaseKernel(kernel);
        }
        if (program) {
            clReleaseProgram(program);
        }
        if (queue) {
            clReleaseCommandQueue(queue);
        }
        if (context) {
            clReleaseContext(context);
        }
    }

    void Initialize(int platform) {
        cl_uint count = 0;
        Check(clGetPlatformIDs(0, nullptr, &count), "clGetPlatformIDs");
        if (platform < 0 || static_cast<cl_uint>(platform) >= count) {
            throw std::out_of_range("OpenCL platform index");
        }
        std::vector<cl_platform_id> platforms(count);
        Check(clGetPlatformIDs(count, platforms.data(), nullptr), "clGetPlatformIDs");
        cl_device_id device = nullptr;
        Check(clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 1, &device, nullptr), "clGetDeviceIDs");
        cl_context_properties properties[] = {
            CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platforms[platform]), 0
        };
        cl_int status = CL_SUCCESS;
        context = clCreateContext(properties, 1, &device, nullptr, nullptr, &status);
        Check(status, "clCreateContext");
        queue = clCreateCommandQueue(context, device, 0, &status);
        Check(status, "clCreateCommandQueue");
        const char* text = source;
        program = clCreateProgramWithSource(context, 1, &text, nullptr, &status);
        Check(status, "clCreateProgramWithSource");
        status = clBuildProgram(program, 1, &device, "-cl-std=CL1.2", nullptr, nullptr);
        if (status != CL_SUCCESS) {
            size_t length = 0;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &length);
            std::string log(length, '\0');
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, length, log.data(), nullptr);
            throw std::runtime_error("clBuildProgram: " + log);
        }
        kernel = clCreateKernel(program, "gelu", &status);
        Check(status, "clCreateKernel");
    }

    void Reserve(size_t bytes) {
        if (bytes <= capacity) {
            return;
        }
        cl_int status = CL_SUCCESS;
        cl_mem next = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, nullptr, &status);
        Check(status, "clCreateBuffer");
        if (buffer) {
            clReleaseMemObject(buffer);
        }
        buffer = next;
        capacity = bytes;
    }
};

}

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (platform < 0) {
        throw std::out_of_range("OpenCL platform index");
    }
    if (input.empty()) {
        return {};
    }
    if (input.size() > std::numeric_limits<size_t>::max() / sizeof(float)) {
        throw std::length_error("OpenCL input size");
    }
    thread_local std::map<int, std::unique_ptr<State>> states;
    auto found = states.find(platform);
    if (found == states.end()) {
        auto state = std::make_unique<State>();
        state->Initialize(platform);
        found = states.emplace(platform, std::move(state)).first;
    }
    State& state = *found->second;
    const size_t bytes = input.size() * sizeof(float);
    state.Reserve(bytes);
    auto allocation = std::async(std::launch::async, [count = input.size()] {
        return std::vector<float>(count);
    });
    std::vector<float> output;
    try {
        Check(clEnqueueWriteBuffer(state.queue, state.buffer, CL_FALSE, 0, bytes, input.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer");
        cl_ulong count = static_cast<cl_ulong>(input.size());
        Check(clSetKernelArg(state.kernel, 0, sizeof(state.buffer), &state.buffer), "clSetKernelArg");
        Check(clSetKernelArg(state.kernel, 1, sizeof(count), &count), "clSetKernelArg");
        const size_t local = 256;
        const size_t global = ((input.size() + 3) / 4 + local - 1) / local * local;
        Check(clEnqueueNDRangeKernel(state.queue, state.kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr), "clEnqueueNDRangeKernel");
        Check(clFlush(state.queue), "clFlush");
        output = allocation.get();
        Check(clEnqueueReadBuffer(state.queue, state.buffer, CL_TRUE, 0, bytes, output.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer");
    } catch (...) {
        clFinish(state.queue);
        throw;
    }
    return output;
}
