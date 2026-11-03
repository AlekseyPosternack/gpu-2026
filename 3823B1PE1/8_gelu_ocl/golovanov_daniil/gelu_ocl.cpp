#include "gelu_ocl.h"

#include <CL/cl.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace {

const char* kKernelSource = R"CLC(
__kernel void gelu_kernel(__global const float* input,
                          __global float* output,
                          const int n)
{
    const int i = get_global_id(0);
    if (i >= n) return;

    const float x = input[i];
    const float x3 = x * x * x;
    const float z = 0.7978845608028654f * (x + 0.044715f * x3);
    const float e = exp(2.0f * z);
    const float t = 1.0f - 2.0f / (e + 1.0f);
    output[i] = 0.5f * x * (1.0f + t);
}
)CLC";

struct OclContext {
    cl_platform_id platform = nullptr;
    cl_device_id   device   = nullptr;
    cl_context     context  = nullptr;
    cl_command_queue queue  = nullptr;
    cl_program     program  = nullptr;
    cl_kernel      kernel   = nullptr;
    cl_mem         in_buf   = nullptr;
    cl_mem         out_buf  = nullptr;
    std::size_t    capacity = 0;

    void init(int platform_index) {
        cl_uint num_platforms = 0;
        if (clGetPlatformIDs(0, nullptr, &num_platforms) != CL_SUCCESS || num_platforms == 0)
            throw std::runtime_error("no OpenCL platforms");
        if (platform_index < 0 || static_cast<cl_uint>(platform_index) >= num_platforms)
            throw std::runtime_error("invalid platform index");

        cl_platform_id platforms[16];
        const cl_uint to_get = num_platforms < 16 ? num_platforms : 16;
        if (clGetPlatformIDs(to_get, platforms, nullptr) != CL_SUCCESS)
            throw std::runtime_error("clGetPlatformIDs failed");
        platform = platforms[platform_index];

        cl_int err = CL_SUCCESS;
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr) != CL_SUCCESS)
            throw std::runtime_error("no GPU device on this platform");

        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("clCreateContext failed");

        queue = clCreateCommandQueue(context, device, 0, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("clCreateCommandQueue failed");

        const char* src = kKernelSource;
        const std::size_t len = std::char_traits<char>::length(src);
        program = clCreateProgramWithSource(context, 1, &src, &len, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("clCreateProgramWithSource failed");

        if (clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", nullptr, nullptr) != CL_SUCCESS) {
            std::size_t log_size = 0;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::string log(log_size, '\0');
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, &log[0], nullptr);
            throw std::runtime_error("clBuildProgram failed: " + log);
        }

        kernel = clCreateKernel(program, "gelu_kernel", &err);
        if (err != CL_SUCCESS) throw std::runtime_error("clCreateKernel failed");
    }

    void ensure_buffers(std::size_t n) {
        if (n <= capacity) return;
        cl_int err = CL_SUCCESS;
        if (in_buf)  { clReleaseMemObject(in_buf);  in_buf = nullptr; }
        if (out_buf) { clReleaseMemObject(out_buf); out_buf = nullptr; }

        in_buf  = clCreateBuffer(context, CL_MEM_READ_ONLY,  n * sizeof(float), nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("clCreateBuffer in failed");
        out_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float), nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("clCreateBuffer out failed");
        capacity = n;
    }

    ~OclContext() {
        if (in_buf)  clReleaseMemObject(in_buf);
        if (out_buf) clReleaseMemObject(out_buf);
        if (kernel)  clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
        if (queue)   clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }
};

OclContext& ctx(int platform) {
    static int cached_platform = -1;
    static OclContext c;
    if (cached_platform != platform) {
        c = OclContext{};
        c.init(platform);
        cached_platform = platform;
    }
    return c;
}

void check(cl_int err, const char* what) {
    if (err != CL_SUCCESS)
        throw std::runtime_error(std::string(what) + " failed: " + std::to_string(err));
}

} // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    const std::size_t n = input.size();
    std::vector<float> output(n);
    if (n == 0) return output;

    OclContext& c = ctx(platform);
    c.ensure_buffers(n);

    check(clEnqueueWriteBuffer(c.queue, c.in_buf, CL_TRUE, 0,
                               n * sizeof(float), input.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer");

    check(clSetKernelArg(c.kernel, 0, sizeof(cl_mem), &c.in_buf),  "clSetKernelArg 0");
    check(clSetKernelArg(c.kernel, 1, sizeof(cl_mem), &c.out_buf), "clSetKernelArg 1");
    const int n_int = static_cast<int>(n);
    check(clSetKernelArg(c.kernel, 2, sizeof(int),    &n_int),     "clSetKernelArg 2");

    const std::size_t local = 256;
    const std::size_t global = ((n + local - 1) / local) * local;
    check(clEnqueueNDRangeKernel(c.queue, c.kernel, 1, nullptr, &global, &local,
                                 0, nullptr, nullptr),
          "clEnqueueNDRangeKernel");

    check(clEnqueueReadBuffer(c.queue, c.out_buf, CL_TRUE, 0,
                              n * sizeof(float), output.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer");

    return output;
}