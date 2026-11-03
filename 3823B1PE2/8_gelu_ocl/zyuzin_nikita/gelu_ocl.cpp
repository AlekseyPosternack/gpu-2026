#include "gelu_ocl.h"

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <cstddef>
#include <thread>

namespace {

const char* kKernelSource = R"(
__kernel void Gelu(__global const float* in, __global float* out, uint n) {
    const float c1 = -1.5957691216057308f;
    const float c2 = c1 * 0.044715f;
    const uint i = get_global_id(0);
    if (i < n) {
        const float x = in[i];
        const float t = fmin(x * (c1 + c2 * x * x), 80.0f);
        out[i] = native_divide(x, 1.0f + native_exp(t));
    }
}
)";

constexpr size_t kLocalSize = 256;

struct Context {
    int platform_index = -1;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_mem in = nullptr;
    cl_mem out = nullptr;
    size_t capacity = 0;

    void Release() {
        if (in) clReleaseMemObject(in);
        if (out) clReleaseMemObject(out);
        if (kernel) clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
        *this = Context();
    }

    void Init(int platform) {
        if (platform == platform_index) {
            return;
        }
        Release();

        cl_uint platform_count = 0;
        clGetPlatformIDs(0, nullptr, &platform_count);
        std::vector<cl_platform_id> platforms(platform_count);
        clGetPlatformIDs(platform_count, platforms.data(), nullptr);

        cl_uint device_count = 0;
        clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count);
        std::vector<cl_device_id> devices(device_count);
        clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, device_count, devices.data(),
                       nullptr);
        cl_device_id device = devices[0];

        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
        queue = clCreateCommandQueue(context, device, 0, nullptr);
        program = clCreateProgramWithSource(context, 1, &kKernelSource, nullptr, nullptr);
        clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", nullptr, nullptr);
        kernel = clCreateKernel(program, "Gelu", nullptr);
        platform_index = platform;
    }

    void Reserve(size_t n) {
        if (n <= capacity) {
            return;
        }
        if (in) clReleaseMemObject(in);
        if (out) clReleaseMemObject(out);
        in = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), nullptr, nullptr);
        out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float), nullptr, nullptr);
        capacity = n;
    }
};

Context& GetContext() {
    static Context ctx;
    return ctx;
}

}  // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    const size_t n = input.size();
    if (n == 0) {
        return {};
    }

    Context& ctx = GetContext();
    ctx.Init(platform);
    ctx.Reserve(n);

    std::vector<float> output;
    std::thread allocator([&output, n] { output = std::vector<float>(n); });

    clEnqueueWriteBuffer(ctx.queue, ctx.in, CL_FALSE, 0, n * sizeof(float), input.data(), 0,
                         nullptr, nullptr);

    const cl_uint count = static_cast<cl_uint>(n);
    clSetKernelArg(ctx.kernel, 0, sizeof(cl_mem), &ctx.in);
    clSetKernelArg(ctx.kernel, 1, sizeof(cl_mem), &ctx.out);
    clSetKernelArg(ctx.kernel, 2, sizeof(cl_uint), &count);
    const size_t global_size = (n + kLocalSize - 1) / kLocalSize * kLocalSize;
    clEnqueueNDRangeKernel(ctx.queue, ctx.kernel, 1, nullptr, &global_size, &kLocalSize, 0,
                           nullptr, nullptr);
    clFlush(ctx.queue);

    allocator.join();

    clEnqueueReadBuffer(ctx.queue, ctx.out, CL_TRUE, 0, n * sizeof(float), output.data(), 0,
                        nullptr, nullptr);

    return output;
}
