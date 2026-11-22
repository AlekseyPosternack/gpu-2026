#include "gelu_ocl.h"

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <stdexcept>
#include <string>

static const char* KERNEL_SRC = R"(
__kernel void gelu_kernel(__global const float* input,
                          __global float* output,
                          const int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    float x  = input[i];
    float x3 = x * x * x;
    float z  = 0.7978845608028654f * (x + 0.044715f * x3); // sqrt(2/pi)*(x+0.044715x^3)
    float e  = exp(2.0f * z);
    float t  = 1.0f - 2.0f / (e + 1.0f);                   // tanh(z)
    output[i] = 0.5f * x * (1.0f + t);
}
)";

namespace {

struct OclState {
    cl_context       context        = nullptr;
    cl_command_queue queue          = nullptr;
    cl_program       program        = nullptr;
    cl_kernel        kernel         = nullptr;
    cl_mem           d_in           = nullptr;
    cl_mem           d_out          = nullptr;
    size_t           capacity_bytes = 0;
    int              platform_id    = -1;
};

OclState g_state;

void Check(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {
        throw std::runtime_error(std::string(msg) + " (err=" +
                                 std::to_string(err) + ")");
    }
}

void ReleaseState() {
    if (g_state.kernel)  { clReleaseKernel(g_state.kernel);          g_state.kernel  = nullptr; }
    if (g_state.program) { clReleaseProgram(g_state.program);        g_state.program = nullptr; }
    if (g_state.queue)   { clReleaseCommandQueue(g_state.queue);     g_state.queue   = nullptr; }
    if (g_state.d_in)    { clReleaseMemObject(g_state.d_in);         g_state.d_in    = nullptr; }
    if (g_state.d_out)   { clReleaseMemObject(g_state.d_out);        g_state.d_out   = nullptr; }
    if (g_state.context) { clReleaseContext(g_state.context);        g_state.context = nullptr; }
    g_state.capacity_bytes = 0;
    g_state.platform_id    = -1;
}

void InitOnce(int platform) {
    if (g_state.context && g_state.platform_id == platform) return;

    // Если платформа сменилась — освобождаем старое состояние
    if (g_state.context) ReleaseState();

    cl_uint num_platforms = 0;
    Check(clGetPlatformIDs(0, nullptr, &num_platforms), "clGetPlatformIDs");
    if (platform < 0 || platform >= static_cast<int>(num_platforms)) {
        throw std::runtime_error("invalid platform index");
    }

    cl_platform_id platforms[16];
    const cl_uint to_get = num_platforms < 16 ? num_platforms : 16;
    Check(clGetPlatformIDs(to_get, platforms, nullptr), "clGetPlatformIDs");
    cl_platform_id pid = platforms[platform];

    cl_device_id device = nullptr;
    Check(clGetDeviceIDs(pid, CL_DEVICE_TYPE_GPU, 1, &device, nullptr),
          "clGetDeviceIDs(GPU)");

    cl_int err = CL_SUCCESS;
    g_state.context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    Check(err, "clCreateContext");

    g_state.queue = clCreateCommandQueue(g_state.context, device, 0, &err);
    Check(err, "clCreateCommandQueue");

    const char* src = KERNEL_SRC;
    g_state.program = clCreateProgramWithSource(g_state.context, 1, &src, nullptr, &err);
    Check(err, "clCreateProgramWithSource");

    // -cl-fast-relaxed-math позволяет заменить exp на быструю аппаратную версию.
    // Для tanh-аппроксимации точности хватает с запасом.
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
    if (g_state.d_in)  { clReleaseMemObject(g_state.d_in);  g_state.d_in  = nullptr; }
    if (g_state.d_out) { clReleaseMemObject(g_state.d_out); g_state.d_out = nullptr; }

    cl_int err = CL_SUCCESS;
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
    const size_t n = input.size();
    std::vector<float> output(n);
    if (n == 0) return output;

    InitOnce(platform);

    const size_t bytes = n * sizeof(float);
    EnsureBuffers(bytes);

    // H2D
    Check(clEnqueueWriteBuffer(g_state.queue, g_state.d_in, CL_TRUE,
                               0, bytes, input.data(),
                               0, nullptr, nullptr),
          "clEnqueueWriteBuffer(H2D)");

    cl_int n_arg = static_cast<cl_int>(n);
    Check(clSetKernelArg(g_state.kernel, 0, sizeof(cl_mem), &g_state.d_in),  "clSetKernelArg(0)");
    Check(clSetKernelArg(g_state.kernel, 1, sizeof(cl_mem), &g_state.d_out), "clSetKernelArg(1)");
    Check(clSetKernelArg(g_state.kernel, 2, sizeof(cl_int), &n_arg),         "clSetKernelArg(2)");

    const size_t local  = 256;
    const size_t global = ((n + local - 1) / local) * local;
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