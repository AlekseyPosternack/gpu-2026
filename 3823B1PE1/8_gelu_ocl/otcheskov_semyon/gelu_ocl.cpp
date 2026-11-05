#include "gelu_ocl.h"

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#define OCL_CHECK(call)                                                      \
    do {                                                                     \
        if (cl_int ocl_status = (call); ocl_status != CL_SUCCESS) {          \
            std::fprintf(stderr, "OpenCL error at %s:%d: code %d\n",         \
                         __FILE__, __LINE__, ocl_status);                    \
            throw std::runtime_error("OpenCL error");                        \
        }                                                                    \
    } while (0)

namespace {

const char* kGeluKernelSource = R"CLC(
__kernel void gelu_vec4(__global const float* input,
                        __global float* output,
                        const unsigned int element_count)
{
    unsigned int base = get_global_id(0) * 4u;
    if (base + 3u < element_count) {
        float4 x = vload4(0, input + base);
        float4 x3 = x * x * x;
        float4 arg = -1.5957691216057308f * (x + 0.044715f * x3);
        vstore4(x / ((float4)(1.0f) + exp(arg)), 0, output + base);
    } else {
        for (unsigned int i = base; i < element_count; ++i) {
            float x = input[i];
            float arg = -1.5957691216057308f * (x + 0.044715f * x * x * x);
            output[i] = x / (1.0f + exp(arg));
        }
    }
}
)CLC";

constexpr std::size_t kLocalWorkSize = 256;
constexpr std::size_t kVectorWidth = 4;
constexpr const char* kKernelName = "gelu_vec4";

struct OpenClContext {
    cl_command_queue command_queue = nullptr;
    cl_context context = nullptr;
    cl_kernel kernel = nullptr;
    cl_mem input_buffer = nullptr;
    cl_mem output_buffer = nullptr;
    std::size_t buffer_element_capacity = 0;
};

OpenClContext create_opencl_context(int platform) {
    cl_uint platform_count = 0;
    OCL_CHECK(clGetPlatformIDs(0, nullptr, &platform_count));
    if (platform < 0 || static_cast<cl_uint>(platform) >= platform_count)
        throw std::runtime_error("Invalid OpenCL platform index");

    std::vector<cl_platform_id> platforms(platform_count);
    OCL_CHECK(clGetPlatformIDs(platform_count, platforms.data(), nullptr));

    cl_device_id device = nullptr;
    OCL_CHECK(clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU,
                             1, &device, nullptr));

    OpenClContext ctx;
    cl_int status = CL_SUCCESS;

    ctx.context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
    OCL_CHECK(status);

#if defined(CL_VERSION_2_0)
    const cl_queue_properties properties[] = {0};
    ctx.command_queue = clCreateCommandQueueWithProperties(
        ctx.context, device, properties, &status);
#else
    ctx.command_queue =
        clCreateCommandQueue(ctx.context, device, 0, &status);
#endif
    OCL_CHECK(status);

    cl_program program = clCreateProgramWithSource(
        ctx.context, 1, &kGeluKernelSource, nullptr, &status);
    OCL_CHECK(status);

    if (clBuildProgram(program, 1, &device, "", nullptr, nullptr) != CL_SUCCESS) {
        std::size_t log_size = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              0, nullptr, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              log_size, log.data(), nullptr);
        std::fprintf(stderr, "OpenCL build failed:\n%s\n", log.c_str());
        throw std::runtime_error("OpenCL build error");
    }

    ctx.kernel = clCreateKernel(program, kKernelName, &status);
    OCL_CHECK(status);
    clReleaseProgram(program);

    return ctx;
}

OpenClContext& get_or_create_context(int platform) {
    static std::map<int, OpenClContext> cached_contexts;

    const auto cached = cached_contexts.find(platform);
    if (cached != cached_contexts.end())
        return cached->second;

    return cached_contexts.emplace(platform, create_opencl_context(platform))
        .first->second;
}

void ensure_buffer_capacity(OpenClContext& ctx, std::size_t element_count) {
    if (ctx.buffer_element_capacity >= element_count)
        return;

    if (ctx.input_buffer)
        clReleaseMemObject(ctx.input_buffer);

    if (ctx.output_buffer)
        clReleaseMemObject(ctx.output_buffer);

    const std::size_t buffer_size_bytes = element_count * sizeof(float);
    cl_int status = CL_SUCCESS;

    ctx.input_buffer = clCreateBuffer(
        ctx.context, CL_MEM_READ_ONLY, buffer_size_bytes, nullptr, &status);
    OCL_CHECK(status);

    ctx.output_buffer = clCreateBuffer(
        ctx.context, CL_MEM_WRITE_ONLY, buffer_size_bytes, nullptr, &status);
    OCL_CHECK(status);

    ctx.buffer_element_capacity = element_count;
}

void enqueue_gelu_kernel(OpenClContext& ctx,
                         const std::vector<float>& input,
                         std::size_t element_count) {
    OCL_CHECK(clEnqueueWriteBuffer(ctx.command_queue, ctx.input_buffer,
                                   CL_FALSE, 0, element_count * sizeof(float),
                                   input.data(), 0, nullptr, nullptr));

    OCL_CHECK(clSetKernelArg(ctx.kernel, 0, sizeof(cl_mem), &ctx.input_buffer));
    OCL_CHECK(clSetKernelArg(ctx.kernel, 1, sizeof(cl_mem), &ctx.output_buffer));
    const unsigned int element_count_arg =
        static_cast<unsigned int>(element_count);
    OCL_CHECK(clSetKernelArg(ctx.kernel, 2, sizeof(unsigned int),
                             &element_count_arg));

    const std::size_t vector_group_count =
        (element_count + kVectorWidth - 1) / kVectorWidth;
    const std::size_t global_work_size =
        ((vector_group_count + kLocalWorkSize - 1) / kLocalWorkSize) *
        kLocalWorkSize;

    OCL_CHECK(clEnqueueNDRangeKernel(ctx.command_queue, ctx.kernel,
                                     1, nullptr, &global_work_size,
                                     &kLocalWorkSize, 0, nullptr, nullptr));
}

} // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    const std::size_t element_count = input.size();
    if (element_count == 0)
        return {};

    OpenClContext& ctx = get_or_create_context(platform);
    ensure_buffer_capacity(ctx, element_count);
    enqueue_gelu_kernel(ctx, input, element_count);

    std::vector<float> output(element_count);
    OCL_CHECK(clEnqueueReadBuffer(ctx.command_queue, ctx.output_buffer,
                                  CL_TRUE, 0, element_count * sizeof(float),
                                  output.data(), 0, nullptr, nullptr));

    return output;
}