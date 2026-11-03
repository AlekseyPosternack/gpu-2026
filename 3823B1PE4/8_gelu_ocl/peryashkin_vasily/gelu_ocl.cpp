
#define CL_TARGET_OPENCL_VERSION 120

#include "gelu_ocl.h"
#include <CL/cl.h>

#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

    const char* KERNEL_SOURCE = R"CLC(
__kernel void gelu_kernel(__global float* data, const ulong n) {
    size_t i = get_global_id(0);
    if (i >= n) return;

    float x = data[i];
    float z = 0.7978845608028654f *
              (x + 0.044715f * x * x * x);

    data[i] = x / (1.0f + exp(-2.0f * z));
}
)CLC";

    void Check(cl_int error, const char* operation) {
        if (error != CL_SUCCESS)
            throw std::runtime_error(
                std::string(operation) + ": " + std::to_string(error)
            );
    }

    struct OpenCLState {
        int platform_index;

        cl_context context = nullptr;
        cl_command_queue queue = nullptr;
        cl_program program = nullptr;
        cl_kernel kernel = nullptr;
        cl_mem buffer = nullptr;

        size_t buffer_capacity = 0;

        explicit OpenCLState(int platform) : platform_index(platform) {
            try {
                cl_int err;
                cl_uint count = 0;

                Check(
                    clGetPlatformIDs(0, nullptr, &count),
                    "clGetPlatformIDs"
                );

                if (platform < 0 || static_cast<cl_uint>(platform) >= count)
                    throw std::invalid_argument("Invalid platform");

                std::vector<cl_platform_id> platforms(count);

                Check(
                    clGetPlatformIDs(count, platforms.data(), nullptr),
                    "clGetPlatformIDs"
                );

                cl_device_id device = nullptr;

                Check(
                    clGetDeviceIDs(
                        platforms[platform],
                        CL_DEVICE_TYPE_GPU,
                        1,
                        &device,
                        nullptr
                    ),
                    "clGetDeviceIDs"
                );

                context = clCreateContext(
                    nullptr, 1, &device, nullptr, nullptr, &err
                );
                Check(err, "clCreateContext");

                queue = clCreateCommandQueue(context, device, 0, &err);
                Check(err, "clCreateCommandQueue");

                program = clCreateProgramWithSource(
                    context, 1, &KERNEL_SOURCE, nullptr, &err
                );
                Check(err, "clCreateProgramWithSource");

                err = clBuildProgram(
                    program, 1, &device, nullptr, nullptr, nullptr
                );

                if (err != CL_SUCCESS) {
                    size_t log_size = 0;

                    clGetProgramBuildInfo(
                        program, device, CL_PROGRAM_BUILD_LOG,
                        0, nullptr, &log_size
                    );

                    std::string log(log_size, '\0');

                    if (log_size)
                        clGetProgramBuildInfo(
                            program, device, CL_PROGRAM_BUILD_LOG,
                            log_size, log.data(), nullptr
                        );

                    throw std::runtime_error("OpenCL build failed:\n" + log);
                }

                kernel = clCreateKernel(program, "gelu_kernel", &err);
                Check(err, "clCreateKernel");

            }
            catch (...) {
                Release();
                throw;
            }
        }

        void EnsureBuffer(size_t bytes) {
            if (buffer && bytes <= buffer_capacity)
                return;

            if (buffer) {
                clReleaseMemObject(buffer);
                buffer = nullptr;
                buffer_capacity = 0;
            }

            cl_int err;

            buffer = clCreateBuffer(
                context, CL_MEM_READ_WRITE, bytes, nullptr, &err
            );
            Check(err, "clCreateBuffer");

            buffer_capacity = bytes;
        }

        void Release() {
            if (buffer) {
                clReleaseMemObject(buffer);
                buffer = nullptr;
            }
            if (kernel) {
                clReleaseKernel(kernel);
                kernel = nullptr;
            }
            if (program) {
                clReleaseProgram(program);
                program = nullptr;
            }
            if (queue) {
                clReleaseCommandQueue(queue);
                queue = nullptr;
            }
            if (context) {
                clReleaseContext(context);
                context = nullptr;
            }

            buffer_capacity = 0;
        }

        ~OpenCLState() {
            Release();
        }
    };

}

std::vector<float> GeluOCL(
    const std::vector<float>& input,
    int platform
) {
    if (input.empty())
        return {};

    static std::mutex mutex;
    static std::unique_ptr<OpenCLState> state;

    std::lock_guard<std::mutex> lock(mutex);

    if (!state || state->platform_index != platform)
        state = std::make_unique<OpenCLState>(platform);

    size_t n = input.size();
    size_t bytes = n * sizeof(float);

    state->EnsureBuffer(bytes);

    Check(
        clEnqueueWriteBuffer(
            state->queue, state->buffer, CL_FALSE,
            0, bytes, input.data(),
            0, nullptr, nullptr
        ),
        "clEnqueueWriteBuffer"
    );

    cl_ulong count = static_cast<cl_ulong>(n);

    Check(
        clSetKernelArg(
            state->kernel, 0, sizeof(cl_mem), &state->buffer
        ),
        "clSetKernelArg"
    );

    Check(
        clSetKernelArg(
            state->kernel, 1, sizeof(cl_ulong), &count
        ),
        "clSetKernelArg"
    );

    const size_t local_size = 256;
    const size_t global_size =
        ((n + local_size - 1) / local_size) * local_size;

    Check(
        clEnqueueNDRangeKernel(
            state->queue, state->kernel, 1,
            nullptr, &global_size, &local_size,
            0, nullptr, nullptr
        ),
        "clEnqueueNDRangeKernel"
    );

    std::vector<float> result(n);

    Check(
        clEnqueueReadBuffer(
            state->queue, state->buffer, CL_TRUE,
            0, bytes, result.data(),
            0, nullptr, nullptr
        ),
        "clEnqueueReadBuffer"
    );

    return result;
}
