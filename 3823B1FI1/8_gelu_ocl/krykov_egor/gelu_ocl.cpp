#include "gelu_ocl.h"

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <cstdio>
#include <cstdlib>
#include <thread>

#define CL_CHECK(call)                                                    \
    do {                                                                  \
        cl_int err_ = (call);                                             \
        if (err_ != CL_SUCCESS) {                                         \
            std::fprintf(stderr, "OpenCL error %d at %s:%d\n",            \
                         static_cast<int>(err_), __FILE__, __LINE__);     \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

namespace {

const char* kKernelSource = R"CLC(
__kernel void gelu(__global const float* in, __global float* out, const uint n) {
    const float k = 2.0f * 0.7978845608028654f;
    const float c = 0.044715f;

    const size_t i = get_global_id(0);
    if (i < n) {
        const float x = in[i];
        const float z = k * (x + c * x * x * x);
        out[i] = x - x / (exp(z) + 1.0f);
    }
}
)CLC";

constexpr size_t kLocalSize = 256;


struct Context {
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_mem in = nullptr;
    cl_mem out = nullptr;
    size_t capacity = 0;

    void Init(int platform_index) {
        cl_uint platform_count = 0;
        CL_CHECK(clGetPlatformIDs(0, nullptr, &platform_count));
        if (platform_index < 0 || static_cast<cl_uint>(platform_index) >= platform_count) {
            std::fprintf(stderr, "OpenCL platform %d not found (available: %u)\n",
                         platform_index, platform_count);
            std::exit(EXIT_FAILURE);
        }
        std::vector<cl_platform_id> platforms(platform_count);
        CL_CHECK(clGetPlatformIDs(platform_count, platforms.data(), nullptr));

        cl_device_id device = nullptr;
        CL_CHECK(clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_GPU, 1, &device, nullptr));

        cl_int err = CL_SUCCESS;
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        CL_CHECK(err);
        queue = clCreateCommandQueue(context, device, 0, &err);
        CL_CHECK(err);

        program = clCreateProgramWithSource(context, 1, &kKernelSource, nullptr, &err);
        CL_CHECK(err);
        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size = 0;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::vector<char> log(log_size + 1, 0);
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            std::fprintf(stderr, "Build failed:\n%s\n", log.data());
            std::exit(EXIT_FAILURE);
        }
        kernel = clCreateKernel(program, "gelu", &err);
        CL_CHECK(err);
    }

    void Reserve(size_t count) {
        if (count <= capacity) return;
        if (in) clReleaseMemObject(in);
        if (out) clReleaseMemObject(out);
        cl_int err = CL_SUCCESS;
        in = clCreateBuffer(context, CL_MEM_READ_ONLY, count * sizeof(float), nullptr, &err);
        CL_CHECK(err);
        out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, count * sizeof(float), nullptr, &err);
        CL_CHECK(err);
        capacity = count;
    }

    ~Context() {
        if (in) clReleaseMemObject(in);
        if (out) clReleaseMemObject(out);
        if (kernel) clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }
};

Context g_ctx;
bool g_initialized = false;

}  // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (input.empty()) {
        return {};
    }
    if (!g_initialized) {
        g_ctx.Init(platform);
        g_initialized = true;
    }

    const size_t n = input.size();
    const size_t bytes = n * sizeof(float);
    g_ctx.Reserve(n);

    std::vector<float> result;
    std::thread alloc_thread([&result, n] { result.resize(n); });

    CL_CHECK(clEnqueueWriteBuffer(g_ctx.queue, g_ctx.in, CL_TRUE, 0, bytes, input.data(),
                                  0, nullptr, nullptr));

    const cl_uint n_arg = static_cast<cl_uint>(n);
    CL_CHECK(clSetKernelArg(g_ctx.kernel, 0, sizeof(cl_mem), &g_ctx.in));
    CL_CHECK(clSetKernelArg(g_ctx.kernel, 1, sizeof(cl_mem), &g_ctx.out));
    CL_CHECK(clSetKernelArg(g_ctx.kernel, 2, sizeof(cl_uint), &n_arg));

    const size_t global = (n + kLocalSize - 1) / kLocalSize * kLocalSize;
    CL_CHECK(clEnqueueNDRangeKernel(g_ctx.queue, g_ctx.kernel, 1, nullptr, &global, &kLocalSize,
                                    0, nullptr, nullptr));
    CL_CHECK(clFlush(g_ctx.queue));

    alloc_thread.join();

    CL_CHECK(clEnqueueReadBuffer(g_ctx.queue, g_ctx.out, CL_TRUE, 0, bytes, result.data(),
                                 0, nullptr, nullptr));

    return result;
}