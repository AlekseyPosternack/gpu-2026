#include "gelu_ocl.h"

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <algorithm>
#include <cstddef>

namespace {

const char* kKernelSource = R"CLC(
#define GELU_A (2.0f * 0.7978845608028654f)
#define GELU_B (GELU_A * 0.044715f)

// i — абсолютный индекс float4 (учитывает global_work_offset), end4 — конец чанка
__kernel void Gelu(__global const float4* in, __global float4* out, uint end4) {
    const uint i = get_global_id(0);
    if (i < end4) {
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
constexpr int kQueues = 4;
constexpr size_t kChunk4 = 1u << 18;

struct Context {
    int platform_index = -1;
    cl_context context = nullptr;
    cl_command_queue queues[kQueues] = {};
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_kernel tail_kernel = nullptr;
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
        if (tail_kernel) clReleaseKernel(tail_kernel);
        if (program) clReleaseProgram(program);
        for (auto& q : queues)
            if (q) clReleaseCommandQueue(q);
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
        for (auto& q : queues) {
            q = clCreateCommandQueue(context, device, 0, &err);
            if (err != CL_SUCCESS) return false;
        }
        program = clCreateProgramWithSource(context, 1, &kKernelSource, nullptr, &err);
        if (err != CL_SUCCESS) return false;
        if (clBuildProgram(program, 1, &device, "-cl-mad-enable", nullptr, nullptr) != CL_SUCCESS)
            return false;
        kernel = clCreateKernel(program, "Gelu", &err);
        if (err != CL_SUCCESS) return false;
        tail_kernel = clCreateKernel(program, "GeluTail", &err);
        if (err != CL_SUCCESS) return false;
        platform_index = platform;
        return true;
    }

    bool Reserve(size_t count) {
        if (count <= capacity) return true;
        ReleaseBuffers();
        cl_int e1 = CL_SUCCESS, e2 = CL_SUCCESS;
        in = clCreateBuffer(context, CL_MEM_READ_ONLY, count * sizeof(float), nullptr, &e1);
        out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, count * sizeof(float), nullptr, &e2);
        if (e1 != CL_SUCCESS || e2 != CL_SUCCESS) {
            ReleaseBuffers();
            return false;
        }
        capacity = count;
        return true;
    }
};

}  // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    const size_t n = input.size();
    if (n == 0) return {};

    static Context ctx;
    if (!ctx.Init(platform) || !ctx.Reserve(n)) return {};

    const size_t n4 = n / 4;
    const size_t tail_off = n4 * 4;
    const size_t tail = n - tail_off;
    const size_t chunks = (n4 + kChunk4 - 1) / kChunk4;

    clSetKernelArg(ctx.kernel, 0, sizeof(cl_mem), &ctx.in);
    clSetKernelArg(ctx.kernel, 1, sizeof(cl_mem), &ctx.out);

    for (size_t c = 0; c < chunks; ++c) {
        const size_t off4 = c * kChunk4;
        const size_t cnt4 = std::min(kChunk4, n4 - off4);
        cl_command_queue q = ctx.queues[c % kQueues];

        clEnqueueWriteBuffer(q, ctx.in, CL_FALSE, off4 * 16, cnt4 * 16,
                             input.data() + off4 * 4, 0, nullptr, nullptr);

        const cl_uint end4 = static_cast<cl_uint>(off4 + cnt4);
        clSetKernelArg(ctx.kernel, 2, sizeof(cl_uint), &end4);
        const size_t global = (cnt4 + kWorkGroupSize - 1) / kWorkGroupSize * kWorkGroupSize;
        clEnqueueNDRangeKernel(q, ctx.kernel, 1, &off4, &global, &kWorkGroupSize,
                               0, nullptr, nullptr);
    }

    cl_command_queue tail_q = ctx.queues[chunks % kQueues];
    if (tail > 0) {
        clEnqueueWriteBuffer(tail_q, ctx.in, CL_FALSE, tail_off * 4, tail * 4,
                             input.data() + tail_off, 0, nullptr, nullptr);
        const cl_uint offset = static_cast<cl_uint>(tail_off);
        const cl_uint total = static_cast<cl_uint>(n);
        clSetKernelArg(ctx.tail_kernel, 0, sizeof(cl_mem), &ctx.in);
        clSetKernelArg(ctx.tail_kernel, 1, sizeof(cl_mem), &ctx.out);
        clSetKernelArg(ctx.tail_kernel, 2, sizeof(cl_uint), &offset);
        clSetKernelArg(ctx.tail_kernel, 3, sizeof(cl_uint), &total);
        clEnqueueNDRangeKernel(tail_q, ctx.tail_kernel, 1, nullptr, &tail, nullptr,
                               0, nullptr, nullptr);
    }

    for (auto q : ctx.queues) clFlush(q);

    std::vector<float> output(n);

    for (size_t c = 0; c < chunks; ++c) {
        const size_t off4 = c * kChunk4;
        const size_t cnt4 = std::min(kChunk4, n4 - off4);
        clEnqueueReadBuffer(ctx.queues[c % kQueues], ctx.out, CL_FALSE, off4 * 16, cnt4 * 16,
                            output.data() + off4 * 4, 0, nullptr, nullptr);
    }
    if (tail > 0) {
        clEnqueueReadBuffer(tail_q, ctx.out, CL_FALSE, tail_off * 4, tail * 4,
                            output.data() + tail_off, 0, nullptr, nullptr);
    }

    for (auto q : ctx.queues) clFinish(q);
    return output;
}