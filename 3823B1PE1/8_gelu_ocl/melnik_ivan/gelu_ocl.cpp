#include "gelu_ocl.h"

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL/cl.h>
#include <cstddef>
#include <vector>

namespace {

const char* const kGeluOpenCLSource = R"CLC(
__kernel void gelu_inplace_vec2(__global float* buf, const ulong total) {
    const size_t idx = get_global_id(0) * 2;
    if (idx + 1 < total) {
        const float2 x = vload2(0, buf + idx);
        const float2 p = x * (-1.5957691216057308f - 0.0713548162726009f * x * x);
        vstore2(x / (1.0f + exp(p)), 0, buf + idx);
    } else if (idx < total) {
        const float x = buf[idx];
        const float p = x * (-1.5957691216057308f - 0.0713548162726009f * x * x);
        buf[idx] = x / (1.0f + exp(p));
    }
}
)CLC";

} // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    const std::size_t n = input.size();
    if (n == 0 || platform < 0) {
        return {};
    }

    static thread_local int initialized_platform = -1;
    static thread_local cl_context ocl_ctx = nullptr;
    static thread_local cl_command_queue ocl_queue = nullptr;
    static thread_local cl_program ocl_prog = nullptr;
    static thread_local cl_kernel ocl_kern = nullptr;
    static thread_local cl_mem dev_buf = nullptr;
    static thread_local std::size_t allocated_bytes = 0;

    if (ocl_kern == nullptr || initialized_platform != platform) {
        if (dev_buf) { clReleaseMemObject(dev_buf); dev_buf = nullptr; }
        if (ocl_kern) { clReleaseKernel(ocl_kern); ocl_kern = nullptr; }
        if (ocl_prog) { clReleaseProgram(ocl_prog); ocl_prog = nullptr; }
        if (ocl_queue) { clReleaseCommandQueue(ocl_queue); ocl_queue = nullptr; }
        if (ocl_ctx) { clReleaseContext(ocl_ctx); ocl_ctx = nullptr; }
        allocated_bytes = 0;

        cl_uint num_platforms = 0;
        if (clGetPlatformIDs(0, nullptr, &num_platforms) != CL_SUCCESS ||
            static_cast<cl_uint>(platform) >= num_platforms) {
            return {};
        }

        std::vector<cl_platform_id> platforms(num_platforms);
        clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

        cl_uint num_devices = 0;
        if (clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices) != CL_SUCCESS ||
            num_devices == 0) {
            return {};
        }

        std::vector<cl_device_id> devices(num_devices);
        clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);
        const cl_device_id target_device = devices[0];

        cl_int err = CL_SUCCESS;
        ocl_ctx = clCreateContext(nullptr, 1, &target_device, nullptr, nullptr, &err);
        ocl_queue = clCreateCommandQueue(ocl_ctx, target_device, 0, &err);
        ocl_prog = clCreateProgramWithSource(ocl_ctx, 1, &kGeluOpenCLSource, nullptr, &err);
        clBuildProgram(ocl_prog, 1, &target_device, "-cl-std=CL1.2", nullptr, nullptr);
        ocl_kern = clCreateKernel(ocl_prog, "gelu_inplace_vec2", &err);

        initialized_platform = platform;
    }

    const std::size_t bytes = n * sizeof(float);
    if (bytes > allocated_bytes) {
        if (dev_buf != nullptr) {
            clReleaseMemObject(dev_buf);
        }
        cl_int err = CL_SUCCESS;
        dev_buf = clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE, bytes, nullptr, &err);
        allocated_bytes = bytes;
    }

    clEnqueueWriteBuffer(ocl_queue, dev_buf, CL_FALSE, 0, bytes, input.data(), 0, nullptr, nullptr);

    const cl_ulong total_elems = static_cast<cl_ulong>(n);
    clSetKernelArg(ocl_kern, 0, sizeof(cl_mem), &dev_buf);
    clSetKernelArg(ocl_kern, 1, sizeof(cl_ulong), &total_elems);

    constexpr std::size_t kLocalSize = 256;
    const std::size_t work_items = (n + 1) / 2;
    const std::size_t global_size = ((work_items + kLocalSize - 1) / kLocalSize) * kLocalSize;

    clEnqueueNDRangeKernel(ocl_queue, ocl_kern, 1, nullptr, &global_size, &kLocalSize, 0, nullptr, nullptr);

    clFlush(ocl_queue);

    std::vector<float> output(n);

    clEnqueueReadBuffer(ocl_queue, dev_buf, CL_TRUE, 0, bytes, output.data(), 0, nullptr, nullptr);

    return output;
}
