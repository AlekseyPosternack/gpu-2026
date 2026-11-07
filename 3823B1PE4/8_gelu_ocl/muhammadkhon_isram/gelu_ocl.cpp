
#include "gelu_ocl.h"

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <cstddef>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const char* kKernelSource = R"CLC(
__kernel void Gelu(__global float* data, ulong count) {

    const size_t i = get_global_id(0) * 4;

    if (i + 3 < count) {

        const float4 x = vload4(0, data + i);

        const float4 arg =
            x * (-1.5957691216057308f
            - 0.0713548162726009f * x * x);

        vstore4(x / (1.0f + exp(arg)), 0, data + i);

    } else {

        for (size_t j = i; j < count; ++j) {

            const float x = data[j];

            const float arg =
                x * (-1.5957691216057308f
                - 0.0713548162726009f * x * x);

            data[j] = x / (1.0f + exp(arg));
        }
    }
}
)CLC";

void CheckCL(cl_int error) {
    if (error != CL_SUCCESS) {
        throw std::runtime_error(
            "OpenCL error: " + std::to_string(error)
        );
    }
}

struct Context {

    int platform;

    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_mem buffer = nullptr;

    std::size_t capacity = 0;

    explicit Context(int index) : platform(index) {
        try {
            cl_uint platform_count = 0;

            CheckCL(
                clGetPlatformIDs(
                    0, nullptr, &platform_count
                )
            );

            if (index < 0 ||
                static_cast<cl_uint>(index) >= platform_count) {
                throw std::invalid_argument(
                    "Invalid OpenCL platform"
                );
            }

            std::vector<cl_platform_id> platforms(
                platform_count
            );

            CheckCL(
                clGetPlatformIDs(
                    platform_count,
                    platforms.data(),
                    nullptr
                )
            );

            cl_device_id device = nullptr;

            CheckCL(
                clGetDeviceIDs(
                    platforms[index],
                    CL_DEVICE_TYPE_GPU,
                    1,
                    &device,
                    nullptr
                )
            );

            cl_int error = CL_SUCCESS;

            context = clCreateContext(
                nullptr, 1, &device,
                nullptr, nullptr, &error
            );
            CheckCL(error);

            queue = clCreateCommandQueue(
                context, device, 0, &error
            );
            CheckCL(error);

            program = clCreateProgramWithSource(
                context, 1, &kKernelSource,
                nullptr, &error
            );
            CheckCL(error);

            error = clBuildProgram(
                program, 1, &device,
                "-cl-mad-enable", nullptr, nullptr
            );

            if (error != CL_SUCCESS) {
                std::size_t length = 0;

                CheckCL(
                    clGetProgramBuildInfo(
                        program, device,
                        CL_PROGRAM_BUILD_LOG,
                        0, nullptr, &length
                    )
                );

                std::string log(length, '\0');

                CheckCL(
                    clGetProgramBuildInfo(
                        program, device,
                        CL_PROGRAM_BUILD_LOG,
                        length, log.data(), nullptr
                    )
                );

                throw std::runtime_error(
                    "OpenCL build failed: " + log
                );
            }

            kernel = clCreateKernel(
                program, "Gelu", &error
            );
            CheckCL(error);

        } catch (...) {
            Release();
            throw;
        }
    }

    ~Context() {
        Release();
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

        capacity = 0;
    }

    void Reserve(std::size_t bytes) {
        if (bytes <= capacity) {
            return;
        }

        cl_int error = CL_SUCCESS;

        cl_mem next = clCreateBuffer(
            context,
            CL_MEM_READ_WRITE,
            bytes,
            nullptr,
            &error
        );

        CheckCL(error);

        if (buffer) {
            clReleaseMemObject(buffer);
        }

        buffer = next;
        capacity = bytes;
    }
};

} // namespace

std::vector<float> GeluOCL(
    const std::vector<float>& input,
    int platform
) {
    if (input.empty()) {
        return {};
    }

    thread_local std::unique_ptr<Context> state;

    if (!state || state->platform != platform) {
        state = std::make_unique<Context>(platform);
    }

    const std::size_t n = input.size();
    const std::size_t bytes = n * sizeof(float);

    state->Reserve(bytes);

    const cl_ulong count = static_cast<cl_ulong>(n);

    CheckCL(
        clSetKernelArg(
            state->kernel,
            0,
            sizeof(cl_mem),
            &state->buffer
        )
    );

    CheckCL(
        clSetKernelArg(
            state->kernel,
            1,
            sizeof(count),
            &count
        )
    );

    constexpr std::size_t local = 256;

    const std::size_t work_items = (n + 3) / 4;

    const std::size_t global =
        ((work_items + local - 1) / local) * local;

    auto output = std::async(
        std::launch::async,
        [n]() {
            return std::vector<float>(n);
        }
    );

    CheckCL(
        clEnqueueWriteBuffer(
            state->queue,
            state->buffer,
            CL_FALSE,
            0,
            bytes,
            input.data(),
            0,
            nullptr,
            nullptr
        )
    );

    try {
        CheckCL(
            clEnqueueNDRangeKernel(
                state->queue,
                state->kernel,
                1,
                nullptr,
                &global,
                &local,
                0,
                nullptr,
                nullptr
            )
        );

        CheckCL(clFlush(state->queue));

        std::vector<float> result = output.get();

        CheckCL(
            clEnqueueReadBuffer(
                state->queue,
                state->buffer,
                CL_TRUE,
                0,
                bytes,
                result.data(),
                0,
                nullptr,
                nullptr
            )
        );

        return result;

    } catch (...) {
        clFinish(state->queue);
        throw;
    }
}
