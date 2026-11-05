#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <vector>

#include "gelu_ocl.h"

namespace {
const char* kKernelSource = R"(
__kernel void gelu(__global const float* input, __global float* output, const unsigned int size) {
    int idx = get_global_id(0);
    if (idx < size) {
        float x = input[idx];
        const float kLog2e = 1.44269504089f;
        const float kA = 1.59576912f * kLog2e;
        const float kB = 0.0713548f * kLog2e;
        
        float t = x * (kA + kB * x * x);
        output[idx] = x / (1.0f + exp2(-t));
    }
}
)";

struct OclWorkspace {
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_kernel kernel = nullptr;
    cl_mem d_in = nullptr;
    cl_mem d_out = nullptr;
    size_t capacity = 0;
    bool is_initialized = false;

    void CompileIfNeeded(int platform_idx) {
        if (is_initialized) return;

        cl_uint num_platforms;
        clGetPlatformIDs(0, nullptr, &num_platforms);
        std::vector<cl_platform_id> platforms(num_platforms);
        clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

        cl_platform_id platform = platforms[platform_idx];

        // Device 0, GPU only
        cl_device_id device;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
        queue = clCreateCommandQueue(context, device, 0, nullptr);

        cl_program program = clCreateProgramWithSource(context, 1, &kKernelSource, nullptr, nullptr);
        clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

        kernel = clCreateKernel(program, "gelu", nullptr);
        clReleaseProgram(program);

        is_initialized = true;
    }

    void AllocateIfNeeded(size_t bytes) {
        if (capacity < bytes) {
            if (d_in) clReleaseMemObject(d_in);
            if (d_out) clReleaseMemObject(d_out);
            d_in = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, nullptr, nullptr);
            d_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, nullptr);
            capacity = bytes;
        }
    }
};

OclWorkspace g_workspace;
} // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (input.empty()) {
        return {};
    }

    const size_t size = input.size();
    const size_t bytes = size * sizeof(float);

    g_workspace.CompileIfNeeded(platform);
    g_workspace.AllocateIfNeeded(bytes);

    // CL_FALSE for asynch host-to-device transfer
    clEnqueueWriteBuffer(g_workspace.queue, g_workspace.d_in, CL_FALSE, 0, bytes, input.data(), 0, nullptr, nullptr);

    unsigned int arg_size = static_cast<unsigned int>(size);
    clSetKernelArg(g_workspace.kernel, 0, sizeof(cl_mem), &g_workspace.d_in);
    clSetKernelArg(g_workspace.kernel, 1, sizeof(cl_mem), &g_workspace.d_out);
    clSetKernelArg(g_workspace.kernel, 2, sizeof(unsigned int), &arg_size);

    size_t local_work_size = 256;
    size_t global_work_size = ((size + local_work_size - 1) / local_work_size) * local_work_size;

    clEnqueueNDRangeKernel(g_workspace.queue, g_workspace.kernel, 1, nullptr, &global_work_size, &local_work_size, 0, nullptr, nullptr);

    std::vector<float> result(size);

    // CL_TRUE forces synchronization, waits for the kernel to finish, and reads back data
    clEnqueueReadBuffer(g_workspace.queue, g_workspace.d_out, CL_TRUE, 0, bytes, result.data(), 0, nullptr, nullptr);

    return result;
}
