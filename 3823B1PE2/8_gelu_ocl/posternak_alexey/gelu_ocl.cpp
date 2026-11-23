#include "gelu_ocl.h"

#include <CL/cl.h>

#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

static const char* kernel_source = R"CLC(
__kernel void gelu_kernel(
    __global const float* input,
    __global float* output,
    const int n)
{
    const int i = get_global_id(0);
    const int base = i * 4;

    if (base >= n)
        return;

    const float K1 = -1.5957691216057308f;
    const float K2 = -0.07135481627260025f;

    if (base + 4 <= n)
    {
        const float4 x = vload4(i, input);
        const float4 z = x * (K1 + K2 * x * x);
        vstore4(x / (1.0f + exp(z)), i, output);
    }
    else
    {
        for (int j = base; j < n; ++j)
        {
            const float x = input[j];
            const float z = x * (K1 + K2 * x * x);
            output[j] = x / (1.0f + exp(z));
        }
    }
}
)CLC";

namespace {

cl_device_id SelectDevice(int platform) {
    cl_uint platform_count = 0;
    if (clGetPlatformIDs(0, nullptr, &platform_count) != CL_SUCCESS || platform_count == 0)
        return nullptr;

    std::vector<cl_platform_id> platforms(platform_count);
    clGetPlatformIDs(platform_count, platforms.data(), nullptr);

    std::vector<cl_platform_id> order;
    if (platform >= 0 && static_cast<cl_uint>(platform) < platform_count)
        order.push_back(platforms[platform]);
    for (cl_uint p = 0; p < platform_count; ++p)
        if (order.empty() || order.front() != platforms[p])
            order.push_back(platforms[p]);

    const cl_device_type types[] = {CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_ALL};
    for (cl_platform_id p : order) {
        for (cl_device_type type : types) {
            cl_device_id device = nullptr;
            if (clGetDeviceIDs(p, type, 1, &device, nullptr) == CL_SUCCESS && device)
                return device;
        }
    }
    return nullptr;
}

struct OpenCLState {
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_mem input_buffer = nullptr;
    cl_mem output_buffer = nullptr;
    size_t capacity = 0;
    size_t local_size = 0;

    explicit OpenCLState(int platform) {
        cl_device_id device = SelectDevice(platform);
        if (!device)
            throw std::runtime_error("GeluOCL: no OpenCL device");

        cl_int err = CL_SUCCESS;
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        queue = clCreateCommandQueue(context, device, 0, &err);
        program = clCreateProgramWithSource(context, 1, &kernel_source, nullptr, &err);

        const cl_int build = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        kernel = clCreateKernel(program, "gelu_kernel", &err);

        if (build != CL_SUCCESS || !context || !queue || !program || !kernel)
            throw std::runtime_error("GeluOCL: OpenCL initialization failed");

        size_t max_wg = 0;
        if (clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
                                     sizeof(max_wg), &max_wg, nullptr) == CL_SUCCESS && max_wg > 0) {
            local_size = 1;
            while (local_size < 256 && local_size * 2 <= max_wg)
                local_size *= 2;
        }
    }

    void Reserve(size_t bytes) {
        if (bytes <= capacity)
            return;

        if (input_buffer) clReleaseMemObject(input_buffer);
        if (output_buffer) clReleaseMemObject(output_buffer);
        input_buffer = nullptr;
        output_buffer = nullptr;
        capacity = 0;

        cl_int err;
        input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, nullptr, &err);
        output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
        if (!input_buffer || !output_buffer)
            throw std::runtime_error("GeluOCL: buffer allocation failed");

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

}  // namespace

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
    const size_t work_items = (input.size() + 3) / 4;
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

    const size_t local = state->local_size;
    const size_t global = local
        ? ((work_items + local - 1) / local) * local
        : work_items;

    clEnqueueNDRangeKernel(
        state->queue,
        state->kernel,
        1,
        nullptr,
        &global,
        local ? &local : nullptr,
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
