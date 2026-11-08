#include "gelu_ocl.h"

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr std::size_t kLocalSize = 256;

const char* kKernelSource = R"CLC(
__kernel void gelu_kernel(__global float* data, ulong size) {
    const size_t index = get_global_id(0);
    if (index < size) {
        const float x = data[index];
        const float exponent = -1.5957691216057308f * (x + 0.044715f * x * x * x);
        data[index] = x / (1.0f + native_exp(exponent));
    }
}
)CLC";

void CheckCl(cl_int error, const char* operation) {
    if (error != CL_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with error " + std::to_string(error));
    }
}

class OpenClResources {
public:
    explicit OpenClResources(int platformIndex) {
        try {
            cl_uint platformCount = 0;
            CheckCl(clGetPlatformIDs(0, nullptr, &platformCount), "clGetPlatformIDs");
            if (platformIndex < 0 || static_cast<cl_uint>(platformIndex) >= platformCount) {
                throw std::invalid_argument("OpenCL platform index is out of range");
            }

            std::vector<cl_platform_id> platforms(platformCount);
            CheckCl(clGetPlatformIDs(platformCount, platforms.data(), nullptr), "clGetPlatformIDs");
            CheckCl(clGetDeviceIDs(platforms[platformIndex], CL_DEVICE_TYPE_GPU, 1, &device_, nullptr), "clGetDeviceIDs");

            cl_int error = CL_SUCCESS;
            context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &error);
            CheckCl(error, "clCreateContext");
            queue_ = clCreateCommandQueue(context_, device_, 0, &error);
            CheckCl(error, "clCreateCommandQueue");

            const std::size_t sourceLength = std::char_traits<char>::length(kKernelSource);
            program_ = clCreateProgramWithSource(context_, 1, &kKernelSource, &sourceLength, &error);
            CheckCl(error, "clCreateProgramWithSource");
            error = clBuildProgram(program_, 1, &device_, "-cl-fast-relaxed-math", nullptr, nullptr);
            if (error != CL_SUCCESS) {
                std::size_t logSize = 0;
                clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
                std::vector<char> log(logSize + 1, '\0');
                clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
                throw std::runtime_error(std::string("clBuildProgram failed: ") + log.data());
            }

            kernel_ = clCreateKernel(program_, "gelu_kernel", &error);
            CheckCl(error, "clCreateKernel");
        } catch (...) {
            Release();
            throw;
        }
    }

    ~OpenClResources() {
        Release();
    }

    cl_mem Reserve(std::size_t size) {
        if (size <= capacity_) {
            return buffer_;
        }
        if (buffer_ != nullptr) {
            clReleaseMemObject(buffer_);
            buffer_ = nullptr;
            capacity_ = 0;
        }
        cl_int error = CL_SUCCESS;
        buffer_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, size * sizeof(float), nullptr, &error);
        CheckCl(error, "clCreateBuffer");
        capacity_ = size;
        return buffer_;
    }

    cl_command_queue Queue() const {
        return queue_;
    }

    cl_kernel Kernel() const {
        return kernel_;
    }

private:
    void Release() {
        if (buffer_ != nullptr) clReleaseMemObject(buffer_);
        if (kernel_ != nullptr) clReleaseKernel(kernel_);
        if (program_ != nullptr) clReleaseProgram(program_);
        if (queue_ != nullptr) clReleaseCommandQueue(queue_);
        if (context_ != nullptr) clReleaseContext(context_);
    }

    cl_device_id device_ = nullptr;
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_program program_ = nullptr;
    cl_kernel kernel_ = nullptr;
    cl_mem buffer_ = nullptr;
    std::size_t capacity_ = 0;
};

OpenClResources& GetResources(int platform) {
    static thread_local std::unordered_map<int, std::unique_ptr<OpenClResources>> resources;
    auto position = resources.find(platform);
    if (position == resources.end()) {
        position = resources.emplace(platform, std::make_unique<OpenClResources>(platform)).first;
    }
    return *position->second;
}

}

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (input.empty()) {
        return {};
    }

    OpenClResources& resources = GetResources(platform);
    const std::size_t bytes = input.size() * sizeof(float);
    cl_mem buffer = resources.Reserve(input.size());
    CheckCl(clEnqueueWriteBuffer(resources.Queue(), buffer, CL_FALSE, 0, bytes, input.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer");

    const cl_ulong size = static_cast<cl_ulong>(input.size());
    CheckCl(clSetKernelArg(resources.Kernel(), 0, sizeof(buffer), &buffer), "clSetKernelArg");
    CheckCl(clSetKernelArg(resources.Kernel(), 1, sizeof(size), &size), "clSetKernelArg");

    const std::size_t globalSize = ((input.size() + kLocalSize - 1) / kLocalSize) * kLocalSize;
    CheckCl(clEnqueueNDRangeKernel(resources.Queue(), resources.Kernel(), 1, nullptr, &globalSize, &kLocalSize, 0, nullptr, nullptr), "clEnqueueNDRangeKernel");

    std::vector<float> result(input.size());
    CheckCl(clEnqueueReadBuffer(resources.Queue(), buffer, CL_TRUE, 0, bytes, result.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer");
    return result;
}
