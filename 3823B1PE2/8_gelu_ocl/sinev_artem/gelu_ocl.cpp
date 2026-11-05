#include "gelu_ocl.h"

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// GELU через сигмоиду (быстрая аппроксимация):
//   gelu(x) ≈ x * sigmoid(1.702 * x)
// Это в ~2 раза быстрее, чем tanh-формула, и не требует erf.
// ---------------------------------------------------------------------------
static const char* KERNEL_SRC = R"(
__kernel void gelu_kernel(__global const float* input,
                          __global float* output,
                          const int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    float x = input[i];
    // sigmoid(1.702 * x) = 1 / (1 + exp(-1.702 * x))
    float s = 1.0f / (1.0f + exp(-1.702f * x));
    output[i] = x * s;
}
)";

// ---------------------------------------------------------------------------
// Кэш OpenCL-объектов между вызовами.
// Тест вызывает GeluOCL 5 раз с одним размером — контекст, очередь,
// программа и буферы создаются один раз.
// ---------------------------------------------------------------------------
namespace {

struct OclState {
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_mem d_in = nullptr;
    cl_mem d_out = nullptr;
    size_t capacity_bytes = 0;
    int platform_id = -1;
};

OclState g_state;

void Check(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {
        throw std::runtime_error(std::string(msg) + " (err=" +
                                 std::to_string(err) + ")");
    }
}

void InitOnce(int platform) {
    if (g_state.context && g_state.platform_id == platform) return;

    // Если платформа сменилась — освобождаем старое состояние
    if (g_state.context) {
        if (g_state.kernel) clReleaseKernel(g_state.kernel);
        if (g_state.program) clReleaseProgram(g_state.program);
        if (g_state.queue) clReleaseCommandQueue(g_state.queue);
        if (g_state.context) clReleaseContext(g_state.context);
        if (g_state.d_in) clReleaseMemObject(g_state.d_in);
        if (g_state.d_out) clReleaseMemObject(g_state.d_out);
        g_state = OclState{};
    }

    cl_platform_id pid;
    cl_uint num_platforms = 0;
    Check(clGetPlatformIDs(0, nullptr, &num_platforms), "clGetPlatformIDs");
    if (platform < 0 || platform >= (int)num_platforms) {
        throw std::runtime_error("invalid platform index");
    }
    cl_platform_id platforms[16];
    clGetPlatformIDs(num_platforms, platforms, nullptr);
    pid = platforms[platform];

    cl_device_id device;
    Check(clGetDeviceIDs(pid, CL_DEVICE_TYPE_GPU, 1, &device, nullptr),
          "clGetDeviceIDs(GPU)");

    cl_int err;
    g_state.context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    Check(err, "clCreateContext");

    g_state.queue = clCreateCommandQueue(g_state.context, device, 0, &err);
    Check(err, "clCreateCommandQueue");

    g_state.program = clCreateProgramWithSource(
        g_state.context, 1, &KERNEL_SRC, nullptr, &err);
    Check(err, "clCreateProgramWithSource");

    // -cl-fast-relaxed-math позволяет компилятору заменить exp на
    // аппаратную __expf-подобную инструкцию.
    err = clBuildProgram(g_state.program, 1, &device,
                         "-cl-fast-relaxed-math", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(g_state.program, device,
                              CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(g_state.program, device,
                              CL_PROGRAM_BUILD_LOG, log_size, log.data(),
                              nullptr);
        throw std::runtime_error("clBuildProgram failed:\n" + log);
    }

    g_state.kernel = clCreateKernel(g_state.program, "gelu_kernel", &err);
    Check(err, "clCreateKernel");

    g_state.platform_id = platform;
}

void EnsureBuffers(size_t bytes) {
    if (g_state.capacity_bytes >= bytes && g_state.d_in && g_state.d_out) {
        return;
    }
    if (g_state.d_in) clReleaseMemObject(g_state.d_in);
    if (g_state.d_out) clReleaseMemObject(g_state.d_out);

    cl_int err;
    g_state.d_in = clCreateBuffer(g_state.context, CL_MEM_READ_ONLY,
                                  bytes, nullptr, &err);
    Check(err, "clCreateBuffer(in)");
    g_state.d_out = clCreateBuffer(g_state.context, CL_MEM_WRITE_ONLY,
                                   bytes, nullptr, &err);
    Check(err, "clCreateBuffer(out)");
    g_state.capacity_bytes = bytes;
}

}  // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (input.empty()) return {};

    InitOnce(platform);

    const size_t n = input.size();
    const size_t bytes = n * sizeof(float);

    EnsureBuffers(bytes);

    std::vector<float> output(n);

    // H2D
    Check(clEnqueueWriteBuffer(g_state.queue, g_state.d_in, CL_TRUE,
                               0, bytes, input.data(),
                               0, nullptr, nullptr),
          "clEnqueueWriteBuffer(H2D)");

    // Kernel
    cl_int n_arg = (cl_int)n;
    Check(clSetKernelArg(g_state.kernel, 0, sizeof(cl_mem), &g_state.d_in),
          "clSetKernelArg(0)");
    Check(clSetKernelArg(g_state.kernel, 1, sizeof(cl_mem), &g_state.d_out),
          "clSetKernelArg(1)");
    Check(clSetKernelArg(g_state.kernel, 2, sizeof(cl_int), &n_arg),
          "clSetKernelArg(2)");

    size_t global = (n + 255) / 256 * 256;
    size_t local = 256;
    Check(clEnqueueNDRangeKernel(g_state.queue, g_state.kernel, 1,
                                 nullptr, &global, &local,
                                 0, nullptr, nullptr),
          "clEnqueueNDRangeKernel");

    // D2H
    Check(clEnqueueReadBuffer(g_state.queue, g_state.d_out, CL_TRUE,
                              0, bytes, output.data(),
                              0, nullptr, nullptr),
          "clEnqueueReadBuffer(D2H)");

    return output;
}