#include "gelu_ocl.h"

#include <CL/cl.h>
#include <vector>
#include <map>
#include <memory>
#include <mutex>

static const char* kernel_source = R"CLC(
__kernel void gelu_kernel(
    __global const float4* input,
    __global float4* output,
    const int n4)
{
    const int i = get_global_id(0);
    if (i >= n4)
        return;

    const float K1 = -1.5957691216057308f;
    const float K2 = -0.0713548262452757f;

    float4 x = input[i];
    float4 z = x * fma(K2, x * x, K1);
    z = fmin(z, (float4)(80.0f));
    output[i] = native_divide(x, (float4)(1.0f) + native_exp(z));
}
)CLC";

namespace {

struct OpenCLState {
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_mem input_buffer = nullptr;
    cl_mem output_buffer = nullptr;
    size_t capacity = 0;

    explicit OpenCLState(int platform) {
        cl_int err;
        cl_uint count;

        clGetPlatformIDs(0, nullptr, &count);

        std::vector<cl_platform_id> platforms(count);
        clGetPlatformIDs(count, platforms.data(), nullptr);

        cl_device_id device;
        clGetDeviceIDs(
            platforms[platform],
            CL_DEVICE_TYPE_GPU,
            1,
            &device,
            nullptr
        );

        context = clCreateContext(
            nullptr, 1, &device, nullptr, nullptr, &err
        );

        queue = clCreateCommandQueue(context, device, 0, &err);

        program = clCreateProgramWithSource(
            context, 1, &kernel_source, nullptr, &err
        );

        clBuildProgram(
            program, 1, &device,
            "-cl-fast-relaxed-math",
            nullptr, nullptr
        );

        kernel = clCreateKernel(program, "gelu_kernel", &err);
    }

    void Reserve(size_t bytes) {
        if (bytes <= capacity)
            return;

        if (input_buffer)
            clReleaseMemObject(input_buffer);

        if (output_buffer)
            clReleaseMemObject(output_buffer);

        cl_int err;

        input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, nullptr, &err);
        output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);

        capacity = bytes;
    }

    ~OpenCLState() {
        if (input_buffer) clReleaseMemObject(input_buffer);
        if (output_buffer) clReleaseMemObject(output_buffer);
        if (kernel) clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }
};

}

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (input.empty())
        return {};

    static std::mutex mutex;
    static std::map<int, std::unique_ptr<OpenCLState>> states;

    std::vector<float> output(input.size());

    std::lock_guard<std::mutex> lock(mutex);

    auto& state = states[platform];

    if (!state)
        state = std::make_unique<OpenCLState>(platform);

    const size_t bytes = input.size() * sizeof(float);
    const int n = static_cast<int>(input.size());

    state->Reserve(bytes);

    clEnqueueWriteBuffer(
        state->queue,
        state->input_buffer,
        CL_FALSE,
        0,
        bytes,
        input.data(),
        0,
        nullptr,
        nullptr
    );

    clSetKernelArg(state->kernel, 0, sizeof(cl_mem), &state->input_buffer);
    clSetKernelArg(state->kernel, 1, sizeof(cl_mem), &state->output_buffer);
    clSetKernelArg(state->kernel, 2, sizeof(int), &n);

    const size_t local_size = 256;
    const size_t work_items = (input.size() + 3) / 4;
    const size_t global_size = ((work_items + local_size - 1) / local_size) * local_size;

    clEnqueueNDRangeKernel(
        state->queue,
        state->kernel,
        1,
        nullptr,
        &global_size,
        &local_size,
        0,
        nullptr,
        nullptr
    );

    clEnqueueReadBuffer(
        state->queue,
        state->output_buffer,
        CL_TRUE,
        0,
        bytes,
        output.data(),
        0,
        nullptr,
        nullptr
    );

    return output;
}