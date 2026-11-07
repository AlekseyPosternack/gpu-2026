#include "gelu_ocl.h"

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#ifndef CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif

#ifdef __APPLE__
#ifndef CL_SILENCE_DEPRECATION
#define CL_SILENCE_DEPRECATION
#endif
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace {

constexpr char GeluKernelSource[] = R"CLC(
__kernel void gelu(__global const float* input,
                   __global float* output,
                   const ulong count) {
    const size_t index = get_global_id(0);
    if (index < count) {
        const float x = input[index];
        const float t = 1.5957691216057308f * x * (1.0f + 0.044715f * x * x);
        const float exponential = exp(-fabs(t));
        const float numerator = x >= 0.0f ? 1.0f : exponential;
        output[index] = x * (numerator / (1.0f + exponential));
    }
}
)CLC";

void CheckOpenCLError(const cl_int error, const char* operation) {
    if (error != CL_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + ": OpenCL error " + std::to_string(error)
        );
    }
}

template <auto Release>
struct OpenCLDeleter {
    template <typename Object>
    void operator()(Object* object) const noexcept {
        Release(object);
    }
};

template <typename Handle, auto Release>
using OpenCLObject = std::unique_ptr<
    std::remove_pointer_t<Handle>, OpenCLDeleter<Release>
>;

using Context = OpenCLObject<cl_context, clReleaseContext>;
using CommandQueue = OpenCLObject<cl_command_queue, clReleaseCommandQueue>;
using Program = OpenCLObject<cl_program, clReleaseProgram>;
using Kernel = OpenCLObject<cl_kernel, clReleaseKernel>;
using Buffer = OpenCLObject<cl_mem, clReleaseMemObject>;

std::string GetBuildLog(cl_program program, cl_device_id device) {
    std::size_t length = 0;
    if (clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              0, nullptr, &length) != CL_SUCCESS || length == 0) {
        return {};
    }

    std::string log(length, '\0');
    if (clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              length, log.data(), nullptr) != CL_SUCCESS) {
        return {};
    }
    if (!log.empty() && log.back() == '\0') {
        log.pop_back();
    }
    return log;
}

struct OpenCLState {
    explicit OpenCLState(const int platformIndex) {
        cl_uint platformCount = 0;
        CheckOpenCLError(clGetPlatformIDs(0, nullptr, &platformCount),
                         "Failed to count OpenCL platforms");
        if (static_cast<unsigned int>(platformIndex) >= platformCount) {
            throw std::out_of_range("OpenCL platform index is out of range");
        }

        std::vector<cl_platform_id> platforms(platformCount);
        CheckOpenCLError(clGetPlatformIDs(platformCount, platforms.data(), nullptr),
                         "Failed to get OpenCL platforms");
        const cl_platform_id selectedPlatform = platforms[platformIndex];
        // num_entries=1 выбирает GPU с индексом 0 на указанной платформе.
        CheckOpenCLError(clGetDeviceIDs(selectedPlatform, CL_DEVICE_TYPE_GPU,
                                        1, &device, nullptr),
                         "Failed to get the first OpenCL GPU device");

        const cl_context_properties properties[] = {
            CL_CONTEXT_PLATFORM,
            reinterpret_cast<cl_context_properties>(selectedPlatform),
            0
        };
        cl_int error = CL_SUCCESS;
        context.reset(clCreateContext(properties, 1, &device,
                                       nullptr, nullptr, &error));
        CheckOpenCLError(error, "Failed to create OpenCL context");
        // Очередь без дополнительных свойств выполняет команды по порядку.
        queue.reset(clCreateCommandQueue(context.get(), device, 0, &error));
        CheckOpenCLError(error, "Failed to create OpenCL command queue");

        const char* source = GeluKernelSource;
        program.reset(clCreateProgramWithSource(context.get(), 1, &source,
                                                nullptr, &error));
        CheckOpenCLError(error, "Failed to create OpenCL program");
        error = clBuildProgram(program.get(), 1, &device,
                                "-cl-std=CL1.2", nullptr, nullptr);
        if (error != CL_SUCCESS) {
            throw std::runtime_error(
                "Failed to build OpenCL GELU program: OpenCL error " +
                std::to_string(error) + "\n" + GetBuildLog(program.get(), device)
            );
        }
        kernel.reset(clCreateKernel(program.get(), "gelu", &error));
        CheckOpenCLError(error, "Failed to create OpenCL GELU kernel");
    }

    ~OpenCLState() noexcept {
        if (queue) {
            clFinish(queue.get());
        }
    }

    void Reserve(const std::size_t bytes) {
        if (bytes <= capacity) {
            return;
        }

        cl_int error = CL_SUCCESS;
        Buffer nextInput(clCreateBuffer(context.get(), CL_MEM_READ_ONLY,
                                         bytes, nullptr, &error));
        CheckOpenCLError(error, "Failed to allocate OpenCL input buffer");
        Buffer nextOutput(clCreateBuffer(context.get(), CL_MEM_WRITE_ONLY,
                                          bytes, nullptr, &error));
        CheckOpenCLError(error, "Failed to allocate OpenCL output buffer");
        // Оба старых буфера сохраняются при ошибке выделения новых.
        input.swap(nextInput);
        output.swap(nextOutput);
        capacity = bytes;
    }

    cl_device_id device = nullptr;
    Context context;
    CommandQueue queue;
    Program program;
    Kernel kernel;
    Buffer input;
    Buffer output;
    std::size_t capacity = 0;
};

OpenCLState& GetState(const int platform) {
    // Отдельные kernel/queue/buffers для каждого CPU-потока и платформы.
    thread_local std::map<int, std::unique_ptr<OpenCLState>> states;
    auto position = states.find(platform);
    if (position == states.end()) {
        auto state = std::make_unique<OpenCLState>(platform);
        position = states.emplace(platform, std::move(state)).first;
    }
    return *position->second;
}

} // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, const int platform) {
    if (platform < 0) {
        throw std::invalid_argument("OpenCL platform index must be non-negative");
    }
    if (input.empty()) {
        return {};
    }
    if (input.size() > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::length_error("OpenCL buffer size is too large");
    }

    OpenCLState& state = GetState(platform);
    const std::size_t bytes = input.size() * sizeof(float);
    state.Reserve(bytes);
    // Blocking write позволяет безопасно использовать input после возврата.
    CheckOpenCLError(clEnqueueWriteBuffer(state.queue.get(), state.input.get(),
                                          CL_TRUE, 0, bytes, input.data(),
                                          0, nullptr, nullptr),
                     "Failed to copy input to OpenCL GPU");

    const cl_mem deviceInput = state.input.get();
    const cl_mem deviceOutput = state.output.get();
    const cl_ulong count = static_cast<cl_ulong>(input.size());
    CheckOpenCLError(clSetKernelArg(state.kernel.get(), 0,
                                    sizeof(deviceInput), &deviceInput),
                     "Failed to set OpenCL input argument");
    CheckOpenCLError(clSetKernelArg(state.kernel.get(), 1,
                                    sizeof(deviceOutput), &deviceOutput),
                     "Failed to set OpenCL output argument");
    CheckOpenCLError(clSetKernelArg(state.kernel.get(), 2, sizeof(count), &count),
                     "Failed to set OpenCL size argument");

    const std::size_t globalSize = input.size();
    CheckOpenCLError(clEnqueueNDRangeKernel(state.queue.get(), state.kernel.get(),
                                            1, nullptr, &globalSize, nullptr,
                                            0, nullptr, nullptr),
                     "Failed to enqueue OpenCL GELU kernel");
    // Запускаем очередь до выделения CPU-памяти для перекрытия с работой GPU.
    CheckOpenCLError(clFlush(state.queue.get()), "Failed to flush OpenCL queue");
    std::vector<float> result(input.size());
    CheckOpenCLError(clEnqueueReadBuffer(state.queue.get(), state.output.get(),
                                         CL_TRUE, 0, bytes, result.data(),
                                         0, nullptr, nullptr),
                     "Failed to copy GELU result from OpenCL GPU");
    return result;
}
