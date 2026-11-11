
#define CL_TARGET_OPENCL_VERSION 120

#include "gelu_ocl.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr size_t LOCAL_SIZE = 256;

const char KERNEL_SOURCE[] = R"CLC(
__kernel void gelu_kernel(
    __global float* data,
    const ulong n)
{
    const ulong i = get_global_id(0) * 4;

    if (i + 3 < n) {
        float4 x = vload4(0, data + i);

        float4 exponent = x * (
            -1.5957691216057308f
            - 0.0713548162726009f * x * x
        );

        float4 result = x / (1.0f + native_exp(exponent));

        vstore4(result, 0, data + i);
    }
    else {
        for (ulong j = i; j < n; ++j) {
            float x = data[j];

            float exponent = x * (
                -1.5957691216057308f
                - 0.0713548162726009f * x * x
            );

            data[j] = x / (1.0f + native_exp(exponent));
        }
    }
}
)CLC";


void CheckOpenCL(cl_int error, const char* operation) {
    if (error != CL_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) +
            " failed with OpenCL error: " +
            std::to_string(error)
        );
    }
}


class OpenCLContext {
public:
    OpenCLContext() = default;

    ~OpenCLContext() {
        Release();
    }

    OpenCLContext(const OpenCLContext&) = delete;
    OpenCLContext& operator=(const OpenCLContext&) = delete;

    void Init(int platform_index) {
        if (context != nullptr && current_platform == platform_index) {
            return;
        }

        Release();

        cl_uint platform_count = 0;

        CheckOpenCL(
            clGetPlatformIDs(0, nullptr, &platform_count),
            "clGetPlatformIDs"
        );

        if (platform_count == 0) {
            throw std::runtime_error("No OpenCL platforms found");
        }

        if (platform_index < 0 ||
            static_cast<cl_uint>(platform_index) >= platform_count) {
            throw std::invalid_argument("Invalid OpenCL platform index");
        }

        std::vector<cl_platform_id> platforms(platform_count);

        CheckOpenCL(
            clGetPlatformIDs(platform_count, platforms.data(), nullptr),
            "clGetPlatformIDs"
        );

        cl_device_id device = nullptr;

        CheckOpenCL(
            clGetDeviceIDs(
                platforms[platform_index],
                CL_DEVICE_TYPE_GPU,
                1,
                &device,
                nullptr
            ),
            "clGetDeviceIDs"
        );

        cl_int error = CL_SUCCESS;

        context = clCreateContext(
            nullptr,
            1,
            &device,
            nullptr,
            nullptr,
            &error
        );

        CheckOpenCL(error, "clCreateContext");

        queue = clCreateCommandQueue(
            context,
            device,
            0,
            &error
        );

        CheckOpenCL(error, "clCreateCommandQueue");

        const char* source = KERNEL_SOURCE;

        program = clCreateProgramWithSource(
            context,
            1,
            &source,
            nullptr,
            &error
        );

        CheckOpenCL(error, "clCreateProgramWithSource");

        error = clBuildProgram(
            program,
            1,
            &device,
            "-cl-fast-relaxed-math",
            nullptr,
            nullptr
        );

        if (error != CL_SUCCESS) {
            size_t log_size = 0;

            clGetProgramBuildInfo(
                program,
                device,
                CL_PROGRAM_BUILD_LOG,
                0,
                nullptr,
                &log_size
            );

            std::string log(log_size, '\0');

            clGetProgramBuildInfo(
                program,
                device,
                CL_PROGRAM_BUILD_LOG,
                log_size,
                &log[0],
                nullptr
            );

            throw std::runtime_error(
                "OpenCL build failed:\n" + log
            );
        }

        kernel = clCreateKernel(
            program,
            "gelu_kernel",
            &error
        );

        CheckOpenCL(error, "clCreateKernel");

        current_platform = platform_index;
    }

    void Reserve(size_t n) {
        if (n <= capacity) {
            return;
        }

        if (buffer != nullptr) {
            clReleaseMemObject(buffer);
            buffer = nullptr;
        }

        cl_int error = CL_SUCCESS;

        buffer = clCreateBuffer(
            context,
            CL_MEM_READ_WRITE,
            n * sizeof(float),
            nullptr,
            &error
        );

        CheckOpenCL(error, "clCreateBuffer");

        capacity = n;
    }

    void Release() {
        if (buffer != nullptr) {
            clReleaseMemObject(buffer);
            buffer = nullptr;
        }

        if (kernel != nullptr) {
            clReleaseKernel(kernel);
            kernel = nullptr;
        }

        if (program != nullptr) {
            clReleaseProgram(program);
            program = nullptr;
        }

        if (queue != nullptr) {
            clReleaseCommandQueue(queue);
            queue = nullptr;
        }

        if (context != nullptr) {
            clReleaseContext(context);
            context = nullptr;
        }

        capacity = 0;
        current_platform = -1;
    }

    cl_command_queue GetQueue() const {
        return queue;
    }

    cl_kernel GetKernel() const {
        return kernel;
    }

    cl_mem GetBuffer() const {
        return buffer;
    }

private:
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_mem buffer = nullptr;

    size_t capacity = 0;
    int current_platform = -1;
};


OpenCLContext& GetContext() {
    static OpenCLContext context;
    return context;
}

} // namespace


std::vector<float> GeluOCL(
    const std::vector<float>& input,
    int platform)
{
    const size_t n = input.size();

    if (n == 0) {
        return {};
    }

    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    OpenCLContext& ctx = GetContext();

    ctx.Init(platform);

    ctx.Reserve(n);

    std::vector<float> output(n);

    cl_mem buffer = ctx.GetBuffer();
    cl_command_queue queue = ctx.GetQueue();
    cl_kernel kernel = ctx.GetKernel();

    const size_t bytes = n * sizeof(float);

    CheckOpenCL(
        clEnqueueWriteBuffer(
            queue,
            buffer,
            CL_FALSE,
            0,
            bytes,
            input.data(),
            0,
            nullptr,
            nullptr
        ),
        "clEnqueueWriteBuffer"
    );

    const cl_ulong count = static_cast<cl_ulong>(n);

    CheckOpenCL(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer),
        "clSetKernelArg"
    );

    CheckOpenCL(
        clSetKernelArg(kernel, 1, sizeof(cl_ulong), &count),
        "clSetKernelArg"
    );

    const size_t work_items = (n + 3) / 4;

    const size_t global_size =
        ((work_items + LOCAL_SIZE - 1) / LOCAL_SIZE) * LOCAL_SIZE;

    CheckOpenCL(
        clEnqueueNDRangeKernel(
            queue,
            kernel,
            1,
            nullptr,
            &global_size,
            &LOCAL_SIZE,
            0,
            nullptr,
            nullptr
        ),
        "clEnqueueNDRangeKernel"
    );

    CheckOpenCL(
        clEnqueueReadBuffer(
            queue,
            buffer,
            CL_TRUE,
            0,
            bytes,
            output.data(),
            0,
            nullptr,
            nullptr
        ),
        "clEnqueueReadBuffer"
    );

    return output;
}
