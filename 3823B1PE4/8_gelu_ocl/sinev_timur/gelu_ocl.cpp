#include "gelu_ocl.h"

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

constexpr const char* kGeluKernelSource = R"CLC(
__kernel void gelu_kernel(__global const float* input,
                          __global float* output,
                          ulong size) {
    const ulong index = (ulong)get_global_id(0);

    if (index >= size) {
        return;
    }

    const float x = input[index];
    const float x3 = x * x * x;
    const float z = 0.7978845608028654f * (x + 0.044715f * x3);

    const float tanh_approx = 2.0f / (1.0f + exp(-2.0f * z)) - 1.0f;

    output[index] = 0.5f * x * (1.0f + tanh_approx);
}
)CLC";

void Check(cl_int status, const char* operation) {
    if (status != CL_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) +
            " failed with OpenCL error " +
            std::to_string(status));
    }
}

} // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (platform < 0) {
        throw std::invalid_argument("Platform index must be non-negative");
    }

    std::vector<float> result(input.size());

    if (input.empty()) {
        return result;
    }

    if (input.size() >
        std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::overflow_error("Input size is too large");
    }

    cl_uint platform_count = 0;

    Check(clGetPlatformIDs(0, nullptr, &platform_count),
          "clGetPlatformIDs(count)");

    if (static_cast<cl_uint>(platform) >= platform_count) {
        throw std::invalid_argument("OpenCL platform index is out of range");
    }

    std::vector<cl_platform_id> platforms(platform_count);

    Check(clGetPlatformIDs(platform_count, platforms.data(), nullptr),
          "clGetPlatformIDs");

    cl_device_id device = nullptr;

    Check(clGetDeviceIDs(platforms[platform],
                         CL_DEVICE_TYPE_GPU,
                         1,
                         &device,
                         nullptr),
          "clGetDeviceIDs(GPU)");

    cl_int status = CL_SUCCESS;

    // Create context
    cl_context context = clCreateContext(
        nullptr, 1, &device, nullptr, nullptr, &status);

    Check(status, "clCreateContext");

    auto context_guard = std::unique_ptr<
        typename std::remove_pointer<cl_context>::type,
        decltype(&clReleaseContext)>(context, clReleaseContext);

    // Create command queue
    cl_command_queue queue = clCreateCommandQueue(
        context, device, 0, &status);

    Check(status, "clCreateCommandQueue");

    auto queue_guard = std::unique_ptr<
        typename std::remove_pointer<cl_command_queue>::type,
        decltype(&clReleaseCommandQueue)>(queue, clReleaseCommandQueue);

    // Create program
    const std::size_t source_length =
        std::char_traits<char>::length(kGeluKernelSource);

    const char* source = kGeluKernelSource;

    cl_program program = clCreateProgramWithSource(
        context, 1, &source, &source_length, &status);

    Check(status, "clCreateProgramWithSource");

    auto program_guard = std::unique_ptr<
        typename std::remove_pointer<cl_program>::type,
        decltype(&clReleaseProgram)>(program, clReleaseProgram);

    // Build program
    status = clBuildProgram(
        program, 1, &device, nullptr, nullptr, nullptr);

    if (status != CL_SUCCESS) {
        std::size_t log_size = 0;

        clGetProgramBuildInfo(
            program, device, CL_PROGRAM_BUILD_LOG,
            0, nullptr, &log_size);

        std::string log(log_size, '\0');

        if (log_size > 0) {
            clGetProgramBuildInfo(
                program, device, CL_PROGRAM_BUILD_LOG,
                log_size, log.data(), nullptr);
        }

        throw std::runtime_error(
            "clBuildProgram failed with OpenCL error " +
            std::to_string(status) + "\n" + log);
    }

    // Create kernel
    cl_kernel kernel = clCreateKernel(
        program, "gelu_kernel", &status);

    Check(status, "clCreateKernel");

    auto kernel_guard = std::unique_ptr<
        typename std::remove_pointer<cl_kernel>::type,
        decltype(&clReleaseKernel)>(kernel, clReleaseKernel);

    const std::size_t bytes = input.size() * sizeof(float);

    // Create input buffer
    cl_mem input_buffer = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        bytes,
        const_cast<float*>(input.data()),
        &status);

    Check(status, "clCreateBuffer(input)");

    auto input_guard = std::unique_ptr<
        typename std::remove_pointer<cl_mem>::type,
        decltype(&clReleaseMemObject)>(input_buffer, clReleaseMemObject);

    // Create output buffer
    cl_mem output_buffer = clCreateBuffer(
        context,
        CL_MEM_WRITE_ONLY,
        bytes,
        nullptr,
        &status);

    Check(status, "clCreateBuffer(output)");

    auto output_guard = std::unique_ptr<
        typename std::remove_pointer<cl_mem>::type,
        decltype(&clReleaseMemObject)>(output_buffer, clReleaseMemObject);

    // Set kernel arguments
    const cl_ulong element_count = static_cast<cl_ulong>(input.size());

    Check(clSetKernelArg(
              kernel, 0, sizeof(input_buffer), &input_buffer),
          "clSetKernelArg(input)");

    Check(clSetKernelArg(
              kernel, 1, sizeof(output_buffer), &output_buffer),
          "clSetKernelArg(output)");

    Check(clSetKernelArg(
              kernel, 2, sizeof(element_count), &element_count),
          "clSetKernelArg(size)");

    // Calculate work size
    constexpr std::size_t local_size = 256;

    const std::size_t groups =
        input.size() / local_size +
        (input.size() % local_size != 0);

    if (groups >
        std::numeric_limits<std::size_t>::max() / local_size) {
        throw std::overflow_error("Global work size is too large");
    }

    const std::size_t global_size = groups * local_size;

    // Execute kernel
    Check(clEnqueueNDRangeKernel(
              queue,
              kernel,
              1,
              nullptr,
              &global_size,
              &local_size,
              0,
              nullptr,
              nullptr),
          "clEnqueueNDRangeKernel");

    // Read result
    Check(clEnqueueReadBuffer(
              queue,
              output_buffer,
              CL_TRUE,
              0,
              bytes,
              result.data(),
              0,
              nullptr,
              nullptr),
          "clEnqueueReadBuffer");

    return result;
}