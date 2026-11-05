#include "gelu_ocl.h"

#include <CL/cl.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace {

const char* GELU_KERNEL = R"CLC(
__kernel void gelu_kernel(
    __global const float* input,
    __global float* output,
    const int size) {
    const int i = get_global_id(0);

    if (i < size) {
        const float x = input[i];
        const float x3 = x * x * x;
        const float inner =
            0.7978845608f * (x + 0.044715f * x3);
        output[i] = 0.5f * x * (1.0f + tanh(inner));
    }
}
)CLC";

void CheckOpenCL(cl_int status, const char* operation) {
    if (status != CL_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed with OpenCL error " +
            std::to_string(status));
    }
}

template <typename Handle, cl_int (*Release)(Handle)>
class OpenCLHandle {
public:
    explicit OpenCLHandle(Handle handle = nullptr) : handle_(handle) {}

    ~OpenCLHandle() {
        if (handle_ != nullptr) {
            Release(handle_);
        }
    }

    OpenCLHandle(const OpenCLHandle&) = delete;
    OpenCLHandle& operator=(const OpenCLHandle&) = delete;

    Handle get() const {
        return handle_;
    }

    Handle* address() {
        return &handle_;
    }

private:
    Handle handle_;
};

} // namespace

std::vector<float> GeluOCL(
    const std::vector<float>& input,
    int platform) {
    if (platform < 0) {
        throw std::invalid_argument("Platform index must not be negative");
    }

    if (input.empty()) {
        return {};
    }

    cl_uint platformCount = 0;
    CheckOpenCL(
        clGetPlatformIDs(0, nullptr, &platformCount),
        "clGetPlatformIDs");

    if (static_cast<cl_uint>(platform) >= platformCount) {
        throw std::invalid_argument("Platform index is out of range");
    }

    std::vector<cl_platform_id> platforms(platformCount);
    CheckOpenCL(
        clGetPlatformIDs(platformCount, platforms.data(), nullptr),
        "clGetPlatformIDs");

    cl_uint deviceCount = 0;
    CheckOpenCL(
        clGetDeviceIDs(
            platforms[platform],
            CL_DEVICE_TYPE_GPU,
            0,
            nullptr,
            &deviceCount),
        "clGetDeviceIDs");

    if (deviceCount == 0) {
        throw std::runtime_error("No GPU device found on this platform");
    }

    std::vector<cl_device_id> devices(deviceCount);
    CheckOpenCL(
        clGetDeviceIDs(
            platforms[platform],
            CL_DEVICE_TYPE_GPU,
            deviceCount,
            devices.data(),
            nullptr),
        "clGetDeviceIDs");

    const cl_device_id device = devices[0];

    cl_int status = CL_SUCCESS;

    OpenCLHandle<cl_context, clReleaseContext> context(
        clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status));
    CheckOpenCL(status, "clCreateContext");

    OpenCLHandle<cl_command_queue, clReleaseCommandQueue> queue(
        clCreateCommandQueue(context.get(), device, 0, &status));
    CheckOpenCL(status, "clCreateCommandQueue");

    const char* source = GELU_KERNEL;
    OpenCLHandle<cl_program, clReleaseProgram> program(
        clCreateProgramWithSource(
            context.get(), 1, &source, nullptr, &status));
    CheckOpenCL(status, "clCreateProgramWithSource");

    status = clBuildProgram(program.get(), 1, &device, nullptr, nullptr, nullptr);
    if (status != CL_SUCCESS) {
        std::size_t logSize = 0;
        clGetProgramBuildInfo(
            program.get(), device, CL_PROGRAM_BUILD_LOG,
            0, nullptr, &logSize);

        std::string buildLog(logSize, '\0');
        if (logSize > 0) {
            clGetProgramBuildInfo(
                program.get(), device, CL_PROGRAM_BUILD_LOG,
                logSize, &buildLog[0], nullptr);
        }

        throw std::runtime_error(
            "clBuildProgram failed: " + buildLog);
    }

    OpenCLHandle<cl_kernel, clReleaseKernel> kernel(
        clCreateKernel(program.get(), "gelu_kernel", &status));
    CheckOpenCL(status, "clCreateKernel");

    const std::size_t bytes = input.size() * sizeof(float);

    OpenCLHandle<cl_mem, clReleaseMemObject> deviceInput(
        clCreateBuffer(
            context.get(),
            CL_MEM_READ_ONLY,
            bytes,
            nullptr,
            &status));
    CheckOpenCL(status, "Create input buffer");

    OpenCLHandle<cl_mem, clReleaseMemObject> deviceOutput(
        clCreateBuffer(
            context.get(),
            CL_MEM_WRITE_ONLY,
            bytes,
            nullptr,
            &status));
    CheckOpenCL(status, "Create output buffer");

    CheckOpenCL(
        clEnqueueWriteBuffer(
            queue.get(),
            deviceInput.get(),
            CL_TRUE,
            0,
            bytes,
            input.data(),
            0,
            nullptr,
            nullptr),
        "Copy input to device");

    const int size = static_cast<int>(input.size());

    CheckOpenCL(
        clSetKernelArg(kernel.get(), 0, sizeof(cl_mem), deviceInput.address()),
        "Set input kernel argument");
    CheckOpenCL(
        clSetKernelArg(kernel.get(), 1, sizeof(cl_mem), deviceOutput.address()),
        "Set output kernel argument");
    CheckOpenCL(
        clSetKernelArg(kernel.get(), 2, sizeof(int), &size),
        "Set size kernel argument");

    const std::size_t globalSize = input.size();
    CheckOpenCL(
        clEnqueueNDRangeKernel(
            queue.get(),
            kernel.get(),
            1,
            nullptr,
            &globalSize,
            nullptr,
            0,
            nullptr,
            nullptr),
        "Run GELU kernel");

    std::vector<float> output(input.size());
    CheckOpenCL(
        clEnqueueReadBuffer(
            queue.get(),
            deviceOutput.get(),
            CL_TRUE,
            0,
            bytes,
            output.data(),
            0,
            nullptr,
            nullptr),
        "Copy output to host");

    return output;
}
