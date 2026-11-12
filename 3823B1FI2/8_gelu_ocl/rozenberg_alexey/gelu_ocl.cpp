#include "gelu_ocl.h"

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif

#include <CL/cl.h>

#include <cstring>

static const char* GELU_KERNEL_SOURCE = R"CLC(
__kernel void gelu_vec4(__global const float4* __restrict input,
                        __global float4* __restrict output,
                        const int n_vec4) {
    int gid = get_global_id(0);
    if (gid < n_vec4) {
        float4 x = input[gid];
        float4 x2 = x * x;
        float4 z = -1.595769122f * x * mad(0.044715f, x2, 1.0f);
        output[gid] = x * native_recip(1.0f + native_exp(z));
    }
}

__kernel void gelu_scalar(__global const float* __restrict input,
                          __global float* __restrict output,
                          const int offset,
                          const int n) {
    int idx = offset + get_global_id(0);
    if (idx < n) {
        float x = input[idx];
        float x2 = x * x;
        float z = -1.595769122f * x * mad(0.044715f, x2, 1.0f);
        output[idx] = x * native_recip(1.0f + native_exp(z));
    }
}
)CLC";

namespace {

struct OCLState {
    int initialized_platform = -1;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel_vec4 = nullptr;
    cl_kernel kernel_scalar = nullptr;

    void init(int platform_idx) {
        if (initialized_platform == platform_idx && context != nullptr) {
            return;
        }
        cleanup();

        cl_uint num_platforms = 0;
        clGetPlatformIDs(0, nullptr, &num_platforms);
        std::vector<cl_platform_id> platforms(num_platforms);
        clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

        cl_platform_id plat = platforms[platform_idx];

        cl_device_id device = nullptr;
        // Note: Standard OpenCL flag is CL_DEVICE_TYPE_GPU
        clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

        cl_int err = CL_SUCCESS;
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        queue = clCreateCommandQueue(context, device, 0, &err);

        size_t src_len = std::strlen(GELU_KERNEL_SOURCE);
        program = clCreateProgramWithSource(context, 1, &GELU_KERNEL_SOURCE, &src_len, &err);
        clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math -cl-mad-enable", nullptr, nullptr);

        kernel_vec4 = clCreateKernel(program, "gelu_vec4", &err);
        kernel_scalar = clCreateKernel(program, "gelu_scalar", &err);

        initialized_platform = platform_idx;
    }

    void cleanup() {
        if (kernel_vec4) { clReleaseKernel(kernel_vec4); kernel_vec4 = nullptr; }
        if (kernel_scalar) { clReleaseKernel(kernel_scalar); kernel_scalar = nullptr; }
        if (program) { clReleaseProgram(program); program = nullptr; }
        if (queue) { clReleaseCommandQueue(queue); queue = nullptr; }
        if (context) { clReleaseContext(context); context = nullptr; }
        initialized_platform = -1;
    }

    ~OCLState() {
        cleanup();
    }
};

OCLState& getOCLState(int platform) {
    static OCLState state;
    state.init(platform);
    return state;
}

} // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (input.empty()) {
        return {};
    }

    OCLState& ocl = getOCLState(platform);

    const size_t n = input.size();
    const size_t bytes = n * sizeof(float);
    cl_int err = CL_SUCCESS;

    // Allocate device buffers
    cl_mem d_in = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY, bytes, nullptr, &err);
    cl_mem d_out = clCreateBuffer(ocl.context, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);

    // Asynchronously copy input data to device (CL_FALSE)
    clEnqueueWriteBuffer(ocl.queue, d_in, CL_FALSE, 0, bytes, input.data(), 0, nullptr, nullptr);

    const int n_vec4 = static_cast<int>(n / 4);
    const size_t local_size = 256;

    // Launch vectorized float4 kernel for the bulk of the array
    if (n_vec4 > 0) {
        size_t global_size = ((static_cast<size_t>(n_vec4) + local_size - 1) / local_size) * local_size;
        clSetKernelArg(ocl.kernel_vec4, 0, sizeof(cl_mem), &d_in);
        clSetKernelArg(ocl.kernel_vec4, 1, sizeof(cl_mem), &d_out);
        clSetKernelArg(ocl.kernel_vec4, 2, sizeof(int), &n_vec4);
        clEnqueueNDRangeKernel(ocl.queue, ocl.kernel_vec4, 1, nullptr, &global_size, &local_size, 0, nullptr, nullptr);
    }

    // Handle remaining tail elements (if input.size() is not a multiple of 4)
    const int offset = n_vec4 * 4;
    const int tail = static_cast<int>(n) - offset;
    if (tail > 0) {
        size_t tail_global = static_cast<size_t>(tail);
        int n_int = static_cast<int>(n);
        clSetKernelArg(ocl.kernel_scalar, 0, sizeof(cl_mem), &d_in);
        clSetKernelArg(ocl.kernel_scalar, 1, sizeof(cl_mem), &d_out);
        clSetKernelArg(ocl.kernel_scalar, 2, sizeof(int), &offset);
        clSetKernelArg(ocl.kernel_scalar, 3, sizeof(int), &n_int);
        clEnqueueNDRangeKernel(ocl.queue, ocl.kernel_scalar, 1, nullptr, &tail_global, nullptr, 0, nullptr, nullptr);
    }

    // Flush the queue so the GPU starts executing transfers and kernels immediately
    clFlush(ocl.queue);

    // Overlap host memory allocation/zero-initialization with GPU execution
    std::vector<float> output(n);

    // Blocking read waits for the in-order queue to finish and copies results to host
    clEnqueueReadBuffer(ocl.queue, d_out, CL_TRUE, 0, bytes, output.data(), 0, nullptr, nullptr);

    clReleaseMemObject(d_in);
    clReleaseMemObject(d_out);

    return output;
}