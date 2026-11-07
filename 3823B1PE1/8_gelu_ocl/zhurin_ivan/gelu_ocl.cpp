#include "gelu_ocl.h"

#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef GELU_OCL_DEVICE_TYPE
#define GELU_OCL_DEVICE_TYPE CL_DEVICE_TYPE_GPU
#endif

namespace {

const char* const kKernelSource = R"CLC(
__kernel void gelu(__global float* data, const ulong size) {
    const size_t i = get_global_id(0);
    if (i < size) {
        const float x = data[i];
        const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
        const float t = 1.0f - 2.0f / (exp(2.0f * inner) + 1.0f);
        data[i] = 0.5f * x * (1.0f + t);
    }
}
)CLC";

constexpr size_t kLocalSize = 256;

void Check(cl_int err, const char* what) {
    if (err != CL_SUCCESS) {
        throw std::runtime_error(std::string(what) + " failed with OpenCL error " +
                                 std::to_string(err));
    }
}

struct State {
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_mem buffer = nullptr;
    size_t buffer_bytes = 0;
};

State Init(int platform_index) {
    cl_uint count = 0;
    Check(clGetPlatformIDs(0, nullptr, &count), "clGetPlatformIDs");
    if (platform_index < 0 || static_cast<cl_uint>(platform_index) >= count) {
        throw std::runtime_error("OpenCL platform index is out of range");
    }
    std::vector<cl_platform_id> platforms(count);
    Check(clGetPlatformIDs(count, platforms.data(), nullptr), "clGetPlatformIDs");
    cl_platform_id platform = platforms[platform_index];

    cl_device_id device = nullptr;
    Check(clGetDeviceIDs(platform, GELU_OCL_DEVICE_TYPE, 1, &device, nullptr),
          "clGetDeviceIDs");

    cl_int err = CL_SUCCESS;
    State s;

    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform), 0};
    s.context = clCreateContext(props, 1, &device, nullptr, nullptr, &err);
    Check(err, "clCreateContext");

    s.queue = clCreateCommandQueue(s.context, device, 0, &err);
    Check(err, "clCreateCommandQueue");

    const char* source = kKernelSource;
    s.program = clCreateProgramWithSource(s.context, 1, &source, nullptr, &err);
    Check(err, "clCreateProgramWithSource");

    err = clBuildProgram(s.program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(s.program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr,
                              &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(s.program, device, CL_PROGRAM_BUILD_LOG, log_size,
                              &log[0], nullptr);
        throw std::runtime_error("clBuildProgram failed:\n" + log);
    }

    s.kernel = clCreateKernel(s.program, "gelu", &err);
    Check(err, "clCreateKernel");
    return s;
}

}

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    const size_t size = input.size();
    if (size == 0) {
        return {};
    }
    const size_t bytes = size * sizeof(float);

    static std::mutex mutex;
    static auto* states = new std::map<int, State>();
    std::lock_guard<std::mutex> lock(mutex);

    auto it = states->find(platform);
    if (it == states->end()) {
        it = states->emplace(platform, Init(platform)).first;
    }
    State& s = it->second;

    cl_int err = CL_SUCCESS;
    if (s.buffer_bytes < bytes) {
        if (s.buffer != nullptr) {
            clReleaseMemObject(s.buffer);
            s.buffer = nullptr;
            s.buffer_bytes = 0;
        }
        s.buffer = clCreateBuffer(s.context, CL_MEM_READ_WRITE, bytes, nullptr, &err);
        Check(err, "clCreateBuffer");
        s.buffer_bytes = bytes;
    }

    Check(clEnqueueWriteBuffer(s.queue, s.buffer, CL_FALSE, 0, bytes, input.data(),
                               0, nullptr, nullptr),
          "clEnqueueWriteBuffer");

    const cl_ulong n = static_cast<cl_ulong>(size);
    Check(clSetKernelArg(s.kernel, 0, sizeof(cl_mem), &s.buffer), "clSetKernelArg(0)");
    Check(clSetKernelArg(s.kernel, 1, sizeof(cl_ulong), &n), "clSetKernelArg(1)");

    const size_t local = kLocalSize;
    const size_t global = ((size + local - 1) / local) * local;
    Check(clEnqueueNDRangeKernel(s.queue, s.kernel, 1, nullptr, &global, &local, 0,
                                 nullptr, nullptr),
          "clEnqueueNDRangeKernel");

    std::vector<float> output(size);

    Check(clEnqueueReadBuffer(s.queue, s.buffer, CL_TRUE, 0, bytes, output.data(), 0,
                              nullptr, nullptr),
          "clEnqueueReadBuffer");

    return output;
}
