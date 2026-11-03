#include "gelu_ocl.h"

#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <thread>

namespace {

constexpr size_t kChunk = size_t(1) << 22;
constexpr size_t kGroup = 256;

const char kSource[] = R"CLC(
__kernel void gelu(__global float* data, ulong begin, ulong end) {
    const ulong i = begin + get_global_id(0) * 4;
    if (i + 3 < end) {
        const float4 x = vload4(0, data + i);
        vstore4(x / (1.0f + exp(-x * (1.5957691216057308f + 0.0713548162726009f * x * x))), 0, data + i);
    } else {
        for (ulong j = i; j < end; ++j) {
            const float x = data[j];
            data[j] = x / (1.0f + exp(-x * (1.5957691216057308f + 0.0713548162726009f * x * x)));
        }
    }
}
)CLC";

struct Context {
    int platform = -1;
    cl_context context = nullptr;
    cl_command_queue upload = nullptr;
    cl_command_queue download = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_mem device = nullptr;
    cl_mem host_out = nullptr;
    float* pinned_out = nullptr;
    size_t capacity = 0;

    explicit Context(int index) : platform(index) {
        cl_uint count = 0;
        clGetPlatformIDs(0, nullptr, &count);
        std::vector<cl_platform_id> platforms(count);
        clGetPlatformIDs(count, platforms.data(), nullptr);
        cl_device_id device_id = nullptr;
        clGetDeviceIDs(platforms[index], CL_DEVICE_TYPE_GPU, 1, &device_id, nullptr);
        context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, nullptr);
        upload = clCreateCommandQueue(context, device_id, 0, nullptr);
        download = clCreateCommandQueue(context, device_id, 0, nullptr);
        const char* source = kSource;
        program = clCreateProgramWithSource(context, 1, &source, nullptr, nullptr);
        clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);
        kernel = clCreateKernel(program, "gelu", nullptr);
    }

    ~Context() {
        Release();
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(upload);
        clReleaseCommandQueue(download);
        clReleaseContext(context);
    }

    void Release() {
        if (pinned_out != nullptr) {
            clEnqueueUnmapMemObject(upload, host_out, pinned_out, 0, nullptr, nullptr);
            clFinish(upload);
        }
        if (host_out != nullptr) {
            clReleaseMemObject(host_out);
        }
        if (device != nullptr) {
            clReleaseMemObject(device);
        }
        host_out = device = nullptr;
        pinned_out = nullptr;
        capacity = 0;
    }

    void Reserve(size_t n) {
        if (n <= capacity) {
            return;
        }
        Release();
        const size_t bytes = n * sizeof(float);
        device = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, nullptr, nullptr);
        host_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, bytes, nullptr, nullptr);
        pinned_out = static_cast<float*>(
            clEnqueueMapBuffer(upload, host_out, CL_TRUE, CL_MAP_READ, 0, bytes, 0, nullptr, nullptr, nullptr));
        capacity = n;
    }
};

Context& GetContext(int platform) {
    static Context* context = nullptr;
    if (context == nullptr || context->platform != platform) {
        delete context;
        context = new Context(platform);
    }
    return *context;
}

}

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    const size_t n = input.size();
    std::vector<float> output;
    if (n == 0) {
        return output;
    }

    Context& ctx = GetContext(platform);
    ctx.Reserve(n);
    const size_t global = ((n + 3) / 4 + kGroup - 1) / kGroup * kGroup;
    const cl_ulong count = n;
    const cl_ulong zero = 0;

    if (ctx.pinned_out == nullptr) {
        output.resize(n);
        clEnqueueWriteBuffer(ctx.upload, ctx.device, CL_FALSE, 0, n * sizeof(float), input.data(), 0, nullptr,
                             nullptr);
        clSetKernelArg(ctx.kernel, 0, sizeof(cl_mem), &ctx.device);
        clSetKernelArg(ctx.kernel, 1, sizeof(zero), &zero);
        clSetKernelArg(ctx.kernel, 2, sizeof(count), &count);
        clEnqueueNDRangeKernel(ctx.upload, ctx.kernel, 1, nullptr, &global, &kGroup, 0, nullptr, nullptr);
        clEnqueueReadBuffer(ctx.upload, ctx.device, CL_TRUE, 0, n * sizeof(float), output.data(), 0, nullptr,
                            nullptr);
        return output;
    }

    const size_t chunks = (n + kChunk - 1) / kChunk;
    std::vector<cl_event> computed(chunks);
    std::vector<cl_event> downloaded(chunks);
    std::atomic<size_t> issued{0};

    output.reserve(n);
    std::thread builder([&] {
        for (size_t c = 0; c < chunks; ++c) {
            while (issued.load(std::memory_order_acquire) <= c) {
                std::this_thread::yield();
            }
            clWaitForEvents(1, &downloaded[c]);
            const size_t begin = c * kChunk;
            output.insert(output.end(), ctx.pinned_out + begin, ctx.pinned_out + std::min(n, begin + kChunk));
        }
    });

    for (size_t c = 0; c < chunks; ++c) {
        const cl_ulong begin = c * kChunk;
        const cl_ulong end = std::min(n, c * kChunk + kChunk);
        const size_t bytes = (end - begin) * sizeof(float);
        const size_t items = ((end - begin + 3) / 4 + kGroup - 1) / kGroup * kGroup;

        clEnqueueWriteBuffer(ctx.upload, ctx.device, CL_FALSE, begin * sizeof(float), bytes, input.data() + begin, 0,
                             nullptr, nullptr);
        clSetKernelArg(ctx.kernel, 0, sizeof(cl_mem), &ctx.device);
        clSetKernelArg(ctx.kernel, 1, sizeof(begin), &begin);
        clSetKernelArg(ctx.kernel, 2, sizeof(end), &end);
        clEnqueueNDRangeKernel(ctx.upload, ctx.kernel, 1, nullptr, &items, &kGroup, 0, nullptr, &computed[c]);
        clFlush(ctx.upload);
        clEnqueueReadBuffer(ctx.download, ctx.device, CL_FALSE, begin * sizeof(float), bytes, ctx.pinned_out + begin, 1,
                            &computed[c], &downloaded[c]);
        clFlush(ctx.download);
        issued.store(c + 1, std::memory_order_release);
    }

    builder.join();
    for (size_t c = 0; c < chunks; ++c) {
        clReleaseEvent(computed[c]);
        clReleaseEvent(downloaded[c]);
    }
    return output;
}
