#include "gelu_ocl.h"

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <cstddef>
#include <vector>

namespace {

// 0.5 * x * (1 + tanh(z)) == x / (1 + exp(-2z)), z = sqrt(2/pi) * (x + 0.044715 * x^3)
const char* kKernelSource = R"CLC(
__kernel void gelu(__global const float4* in, __global float4* out,
                   __global const float* in_scalar, __global float* out_scalar,
                   uint vec_count, uint tail_count) {
    const float k1 = -2.0f * 0.7978845608028654f;
    const float k2 = k1 * 0.044715f;
    const uint i = get_global_id(0);
    if (i < vec_count) {
        const float4 x = in[i];
        out[i] = native_divide(x, 1.0f + native_exp(x * (k1 + k2 * x * x)));
    }
    if (i < tail_count) {
        const uint j = vec_count * 4 + i;
        const float x = in_scalar[j];
        out_scalar[j] = native_divide(x, 1.0f + native_exp(x * (k1 + k2 * x * x)));
    }
}
)CLC";

constexpr size_t kLocalSize = 256;

struct OclContext {
    int platform_index = -1;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_mem in = nullptr;
    cl_mem out = nullptr;
    size_t capacity = 0;

    void ReleaseBuffers() {
        if (in) clReleaseMemObject(in);
        if (out) clReleaseMemObject(out);
        in = out = nullptr;
        capacity = 0;
    }

    void Release() {
        ReleaseBuffers();
        if (kernel) clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
        kernel = nullptr;
        program = nullptr;
        queue = nullptr;
        context = nullptr;
        platform_index = -1;
    }

    bool Init(int platform) {
        if (platform == platform_index) {
            return true;
        }
        Release();

        cl_uint platform_count = 0;
        clGetPlatformIDs(0, nullptr, &platform_count);
        if (platform < 0 || static_cast<cl_uint>(platform) >= platform_count) {
            return false;
        }
        std::vector<cl_platform_id> platforms(platform_count);
        clGetPlatformIDs(platform_count, platforms.data(), nullptr);

        cl_uint device_count = 0;
        clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count);
        if (device_count == 0) {
            return false;
        }
        std::vector<cl_device_id> devices(device_count);
        clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, device_count, devices.data(),
                       nullptr);
        cl_device_id device = devices[0];

        cl_int err = CL_SUCCESS;
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        queue = clCreateCommandQueue(context, device, 0, &err);
        program = clCreateProgramWithSource(context, 1, &kKernelSource, nullptr, &err);
        err = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", nullptr, nullptr);
        if (err != CL_SUCCESS) {
            Release();
            return false;
        }
        kernel = clCreateKernel(program, "gelu", &err);
        platform_index = platform;
        return true;
    }

    void Reserve(size_t size) {
        if (size <= capacity) {
            return;
        }
        ReleaseBuffers();
        cl_int err = CL_SUCCESS;
        in = clCreateBuffer(context, CL_MEM_READ_ONLY, size * sizeof(float), nullptr, &err);
        out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size * sizeof(float), nullptr, &err);
        capacity = size;
    }

    ~OclContext() { Release(); }
};

OclContext ctx;

}  // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    const size_t size = input.size();
    if (size == 0 || !ctx.Init(platform)) {
        return {};
    }
    ctx.Reserve(size);
    const size_t bytes = size * sizeof(float);

    clEnqueueWriteBuffer(ctx.queue, ctx.in, CL_FALSE, 0, bytes, input.data(), 0, nullptr,
                         nullptr);

    const cl_uint vec_count = static_cast<cl_uint>(size / 4);
    const cl_uint tail_count = static_cast<cl_uint>(size % 4);

    clSetKernelArg(ctx.kernel, 0, sizeof(cl_mem), &ctx.in);
    clSetKernelArg(ctx.kernel, 1, sizeof(cl_mem), &ctx.out);
    clSetKernelArg(ctx.kernel, 2, sizeof(cl_mem), &ctx.in);
    clSetKernelArg(ctx.kernel, 3, sizeof(cl_mem), &ctx.out);
    clSetKernelArg(ctx.kernel, 4, sizeof(cl_uint), &vec_count);
    clSetKernelArg(ctx.kernel, 5, sizeof(cl_uint), &tail_count);

    const size_t work_items = vec_count > tail_count ? vec_count : tail_count;
    const size_t global_size = (work_items + kLocalSize - 1) / kLocalSize * kLocalSize;
    clEnqueueNDRangeKernel(ctx.queue, ctx.kernel, 1, nullptr, &global_size, &kLocalSize, 0,
                           nullptr, nullptr);

    std::vector<float> output(size);
    clEnqueueReadBuffer(ctx.queue, ctx.out, CL_TRUE, 0, bytes, output.data(), 0, nullptr,
                        nullptr);
    return output;
}
