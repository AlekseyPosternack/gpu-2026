#include "gelu_ocl.h"
#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>
#include <iostream>
#include <stdexcept>
#include <vector>

static const char* kGeluKernelSource = R"(
__kernel void gelu_kernel(__global const float* input, __global float* output, const int n) {
    int idx = get_global_id(0);
    if (idx < n) {
        float x = input[idx];
        float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
        float exp_val = native_exp(2.0f * inner);
        float t = (exp_val - 1.0f) / (exp_val + 1.0f);
        output[idx] = 0.5f * x * (1.0f + t);
    }
}
)";

struct OpenCLWorkspace {
    cl_platform_id platform_id = nullptr;
    cl_device_id device_id = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    
    cl_mem d_input = nullptr;
    cl_mem d_output = nullptr;
    size_t capacity_bytes = 0;
    int active_platform = -1;
    bool initialized = false;

    void init(int platform_index, size_t required_bytes) {
        // Переинициализируем, если сменился индекс платформы или еще не запускались
        if (!initialized || active_platform != platform_index) {
            if (kernel) { clReleaseKernel(kernel); kernel = nullptr; }
            if (program) { clReleaseProgram(program); program = nullptr; }
            if (queue) { clReleaseCommandQueue(queue); queue = nullptr; }
            if (context) { clReleaseContext(context); context = nullptr; }
            if (d_input) { clReleaseMemObject(d_input); d_input = nullptr; }
            if (d_output) { clReleaseMemObject(d_output); d_output = nullptr; }
            capacity_bytes = 0;

            cl_uint num_platforms = 0;
            clGetPlatformIDs(0, nullptr, &num_platforms);
            if (num_platforms == 0) throw std::runtime_error("No OpenCL platforms found.");

            std::vector<cl_platform_id> platforms(num_platforms);
            clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
            
            if (platform_index < 0 || static_cast<cl_uint>(platform_index) >= num_platforms) {
                platform_index = 0;
            }
            platform_id = platforms[platform_index];

            cl_device_id device;
            cl_int err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
            if (err != CL_SUCCESS) {
                clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device, nullptr);
            }
            device_id = device;

            context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, nullptr);
#ifdef CL_VERSION_2_0
            queue = clCreateCommandQueueWithProperties(context, device_id, 0, nullptr);
#else
            queue = clCreateCommandQueue(context, device_id, 0, nullptr);
#endif
            program = clCreateProgramWithSource(context, 1, &kGeluKernelSource, nullptr, nullptr);
            clBuildProgram(program, 1, &device_id, "-cl-fast-relaxed-math", nullptr, nullptr);
            
            kernel = clCreateKernel(program, "gelu_kernel", nullptr);
            active_platform = platform_index;
            initialized = true;
        }

        if (capacity_bytes < required_bytes) {
            if (d_input) clReleaseMemObject(d_input);
            if (d_output) clReleaseMemObject(d_output);

            d_input = clCreateBuffer(context, CL_MEM_READ_ONLY, required_bytes, nullptr, nullptr);
            d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, required_bytes, nullptr, nullptr);
            capacity_bytes = required_bytes;
        }
    }
};

static OpenCLWorkspace g_ws;

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (input.empty()) return {};

    size_t n = input.size();
    size_t bytes = n * sizeof(float);

    g_ws.init(platform, bytes);

    clEnqueueWriteBuffer(g_ws.queue, g_ws.d_input, CL_TRUE, 0, bytes, input.data(), 0, nullptr, nullptr);

    int size_arg = static_cast<int>(n);
    clSetKernelArg(g_ws.kernel, 0, sizeof(cl_mem), &g_ws.d_input);
    clSetKernelArg(g_ws.kernel, 1, sizeof(cl_mem), &g_ws.d_output);
    clSetKernelArg(g_ws.kernel, 2, sizeof(int), &size_arg);

    size_t local_work_size = 256;
    size_t global_work_size = ((n + local_work_size - 1) / local_work_size) * local_work_size;

    clEnqueueNDRangeKernel(g_ws.queue, g_ws.kernel, 1, nullptr, 
                           &global_work_size, &local_work_size, 0, nullptr, nullptr);

    std::vector<float> output(n);
    clEnqueueReadBuffer(g_ws.queue, g_ws.d_output, CL_TRUE, 0, 
                        bytes, output.data(), 0, nullptr, nullptr);

    return output;
}