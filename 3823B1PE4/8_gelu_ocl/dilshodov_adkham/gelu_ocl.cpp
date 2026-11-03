#include "gelu_ocl.h"

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

namespace {

// 0.5 * x * (1 + tanh(z)) == x * sigmoid(u), u = 2z = 2 * sqrt(2/pi) * (x + 0.044715 * x^3).
// Sigmoid is computed in the stable form with exp(-|u|) <= 1, so exp never overflows
// (with an overflowing exp the result for large negative x depends on the driver).
const char* kKernelSource = R"CLC(
#define GELU_A (2.0f * 0.7978845608028654f)
#define GELU_B (GELU_A * 0.044715f)

__kernel void Gelu(__global const float4* in, __global float4* out, uint n4) {
    const uint i = get_global_id(0);
    if (i < n4) {
        const float4 x = in[i];
        const float4 u = x * (GELU_A + GELU_B * x * x);
        const float4 e = exp(-fabs(u));
        const float4 numerator = select(e, (float4)(1.0f), isgreaterequal(u, (float4)(0.0f)));
        out[i] = x * numerator / (1.0f + e);
    }
}

__kernel void GeluTail(__global const float* in, __global float* out, uint offset, uint n) {
    const uint i = offset + get_global_id(0);
    if (i < n) {
        const float x = in[i];
        const float u = x * (GELU_A + GELU_B * x * x);
        const float e = exp(-fabs(u));
        out[i] = x * (u >= 0.0f ? 1.0f : e) / (1.0f + e);
    }
}
)CLC";

constexpr size_t kWorkGroupSize = 256;

// OpenCL boilerplate (context, queue, program, kernels) is done once per platform,
// device buffers are reused while input size fits
struct Context {
    int platform_index = -1;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_kernel tail_kernel = nullptr;
    cl_mem in = nullptr;
    cl_mem out = nullptr;
    size_t capacity = 0;

    void Release() {
        if (in) clReleaseMemObject(in);
        if (out) clReleaseMemObject(out);
        if (kernel) clReleaseKernel(kernel);
        if (tail_kernel) clReleaseKernel(tail_kernel);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
        *this = Context();
    }

    bool Init(int platform) {
        if (platform_index == platform) return true;
        Release();

        cl_uint platform_count = 0;
        clGetPlatformIDs(0, nullptr, &platform_count);
        if (platform < 0 || static_cast<cl_uint>(platform) >= platform_count) return false;
        std::vector<cl_platform_id> platforms(platform_count);
        clGetPlatformIDs(platform_count, platforms.data(), nullptr);

        cl_uint device_count = 0;
        clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count);
        if (device_count == 0) return false;
        std::vector<cl_device_id> devices(device_count);
        clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, device_count, devices.data(), nullptr);
        cl_device_id device = devices[0];

        cl_int err = CL_SUCCESS;
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) return false;
        queue = clCreateCommandQueue(context, device, 0, &err);
        if (err != CL_SUCCESS) return false;
        program = clCreateProgramWithSource(context, 1, &kKernelSource, nullptr, &err);
        if (err != CL_SUCCESS) return false;
        if (clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr) != CL_SUCCESS) {
            return false;
        }
        kernel = clCreateKernel(program, "Gelu", &err);
        if (err != CL_SUCCESS) return false;
        tail_kernel = clCreateKernel(program, "GeluTail", &err);
        if (err != CL_SUCCESS) return false;
        platform_index = platform;
        return true;
    }

    bool Reserve(size_t count) {
        if (count <= capacity) return true;
        if (in) clReleaseMemObject(in);
        if (out) clReleaseMemObject(out);
        cl_int err_in = CL_SUCCESS;
        cl_int err_out = CL_SUCCESS;
        in = clCreateBuffer(context, CL_MEM_READ_ONLY, count * sizeof(float), nullptr, &err_in);
        out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, count * sizeof(float), nullptr, &err_out);
        capacity = (err_in == CL_SUCCESS && err_out == CL_SUCCESS) ? count : 0;
        return capacity != 0;
    }
};

}  // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    const size_t n = input.size();
    if (n == 0) return {};

    static Context ctx;
    if (!ctx.Init(platform) || !ctx.Reserve(n)) return {};

    const size_t bytes = n * sizeof(float);
    clEnqueueWriteBuffer(ctx.queue, ctx.in, CL_FALSE, 0, bytes, input.data(), 0, nullptr, nullptr);

    const cl_uint n4 = static_cast<cl_uint>(n / 4);
    if (n4 > 0) {
        clSetKernelArg(ctx.kernel, 0, sizeof(cl_mem), &ctx.in);
        clSetKernelArg(ctx.kernel, 1, sizeof(cl_mem), &ctx.out);
        clSetKernelArg(ctx.kernel, 2, sizeof(cl_uint), &n4);
        const size_t global = (n4 + kWorkGroupSize - 1) / kWorkGroupSize * kWorkGroupSize;
        clEnqueueNDRangeKernel(ctx.queue, ctx.kernel, 1, nullptr, &global, &kWorkGroupSize, 0, nullptr, nullptr);
    }
    const cl_uint tail_offset = n4 * 4;
    const cl_uint total = static_cast<cl_uint>(n);
    if (tail_offset < total) {
        clSetKernelArg(ctx.tail_kernel, 0, sizeof(cl_mem), &ctx.in);
        clSetKernelArg(ctx.tail_kernel, 1, sizeof(cl_mem), &ctx.out);
        clSetKernelArg(ctx.tail_kernel, 2, sizeof(cl_uint), &tail_offset);
        clSetKernelArg(ctx.tail_kernel, 3, sizeof(cl_uint), &total);
        const size_t global = total - tail_offset;
        clEnqueueNDRangeKernel(ctx.queue, ctx.tail_kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
    }
    clFlush(ctx.queue);

    // GPU work is already queued, so the result is allocated on host while it runs
    std::vector<float> output(n);
    clEnqueueReadBuffer(ctx.queue, ctx.out, CL_TRUE, 0, bytes, output.data(), 0, nullptr, nullptr);
    return output;
}
