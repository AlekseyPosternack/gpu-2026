#define CL_TARGET_OPENCL_VERSION 120

#include "gelu_ocl.h"

#include <CL/cl.h>

#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kLocalSize = 256;

// Same transformation as in the other GELU tasks:
// GELU(x) = 0.5 * x * (1 + tanh(z)), z = sqrt(2/pi) * (x + 0.044715 * x^3)
// Using 0.5 * (1 + tanh(z)) == 1 / (1 + exp(-2z)):
//     GELU(x) = x / (1 + exp(-2 * sqrt(2/pi) * (x + 0.044715 * x^3)))
const char* const kKernelSource = R"CLC(
inline float gelu(float x) {
    const float a = -1.5957691216057308f;
    const float b = a * 0.044715f;
    return x / (1.0f + exp(x * (a + b * x * x)));
}

// In-place processing: a single device buffer is enough.
// Interaction is vectorized with float4 (4 elements per work item).
__kernel void gelu_kernel(__global float* data, const ulong n) {
    const size_t i = get_global_id(0);
    const size_t n4 = n >> 2;
    __global float4* vec = (__global float4*)data;

    if (i < n4) {
        float4 v = vec[i];
        v.x = gelu(v.x);
        v.y = gelu(v.y);
        v.z = gelu(v.z);
        v.w = gelu(v.w);
        vec[i] = v;
    } else if (i - n4 < (n & 3)) {
        const size_t tail = n4 * 4 + (i - n4);
        data[tail] = gelu(data[tail]);
    }
}
)CLC";

void Check(cl_int error, const char* operation) {
    if (error != CL_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed (OpenCL error " +
            std::to_string(error) + ")");
    }
}

// OpenCL resources are created once per platform and reused across calls.
class OpenCLState {
public:
    OpenCLState() = default;
    OpenCLState(const OpenCLState&) = delete;
    OpenCLState& operator=(const OpenCLState&) = delete;

    ~OpenCLState() { Release(); }

    void Init(int platform) {
        cl_uint platform_count = 0;
        Check(clGetPlatformIDs(0, nullptr, &platform_count),
              "clGetPlatformIDs");

        if (platform < 0 ||
            static_cast<cl_uint>(platform) >= platform_count) {
            throw std::invalid_argument(
                "Invalid OpenCL platform index: " +
                std::to_string(platform));
        }

        std::vector<cl_platform_id> platforms(platform_count);
        Check(clGetPlatformIDs(platform_count, platforms.data(), nullptr),
              "clGetPlatformIDs");

        // Device 0 of the selected platform, GPU devices only.
        cl_device_id device = nullptr;
        Check(clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 1,
                             &device, nullptr),
              "clGetDeviceIDs");

        cl_int error = CL_SUCCESS;

        context_ = clCreateContext(nullptr, 1, &device, nullptr, nullptr,
                                   &error);
        Check(error, "clCreateContext");

        queue_ = clCreateCommandQueue(context_, device, 0, &error);
        Check(error, "clCreateCommandQueue");

        const char* sources[] = {kKernelSource};
        program_ = clCreateProgramWithSource(context_, 1, sources, nullptr,
                                             &error);
        Check(error, "clCreateProgramWithSource");

        error = clBuildProgram(program_, 1, &device, "-cl-mad-enable",
                               nullptr, nullptr);
        if (error != CL_SUCCESS) {
            std::size_t log_size = 0;
            clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG, 0,
                                  nullptr, &log_size);

            std::string log(log_size, '\0');
            if (log_size != 0) {
                clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG,
                                      log_size, log.data(), nullptr);
            }

            throw std::runtime_error(
                "OpenCL program build failed:\n" + log);
        }

        kernel_ = clCreateKernel(program_, "gelu_kernel", &error);
        Check(error, "clCreateKernel");
    }

    // Uploads the input, launches the kernel and reads the result back.
    // The write is asynchronous so the caller can allocate the output
    // vector while the transfer and the kernel are running.
    void Submit(const float* input, std::size_t n) {
        const std::size_t bytes = n * sizeof(float);
        EnsureBuffer(bytes);

        cl_int error = clEnqueueWriteBuffer(
            queue_, buffer_, CL_FALSE, 0, bytes, input, 0, nullptr, nullptr);
        Check(error, "clEnqueueWriteBuffer");

        const cl_ulong count = static_cast<cl_ulong>(n);
        Check(clSetKernelArg(kernel_, 0, sizeof(cl_mem), &buffer_),
              "clSetKernelArg");
        Check(clSetKernelArg(kernel_, 1, sizeof(cl_ulong), &count),
              "clSetKernelArg");

        // Four values per work item plus one work item per tail element.
        const std::size_t work = n / 4 + n % 4;
        const std::size_t global_size =
            ((work + kLocalSize - 1) / kLocalSize) * kLocalSize;

        error = clEnqueueNDRangeKernel(queue_, kernel_, 1, nullptr,
                                       &global_size, &kLocalSize, 0, nullptr,
                                       nullptr);
        Check(error, "clEnqueueNDRangeKernel");
    }

    void Await(float* output, std::size_t n) {
        const std::size_t bytes = n * sizeof(float);
        Check(clEnqueueReadBuffer(queue_, buffer_, CL_TRUE, 0, bytes, output,
                                  0, nullptr, nullptr),
              "clEnqueueReadBuffer");
    }

private:
    void EnsureBuffer(std::size_t bytes) {
        if (buffer_ != nullptr && bytes <= capacity_) {
            return;
        }

        if (buffer_ != nullptr) {
            clReleaseMemObject(buffer_);
            buffer_ = nullptr;
            capacity_ = 0;
        }

        cl_int error = CL_SUCCESS;
        buffer_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, bytes, nullptr,
                                 &error);
        Check(error, "clCreateBuffer");
        capacity_ = bytes;
    }

    void Release() {
        if (buffer_ != nullptr) {
            clReleaseMemObject(buffer_);
            buffer_ = nullptr;
        }
        if (kernel_ != nullptr) {
            clReleaseKernel(kernel_);
            kernel_ = nullptr;
        }
        if (program_ != nullptr) {
            clReleaseProgram(program_);
            program_ = nullptr;
        }
        if (queue_ != nullptr) {
            clReleaseCommandQueue(queue_);
            queue_ = nullptr;
        }
        if (context_ != nullptr) {
            clReleaseContext(context_);
            context_ = nullptr;
        }
        capacity_ = 0;
    }

    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_program program_ = nullptr;
    cl_kernel kernel_ = nullptr;
    cl_mem buffer_ = nullptr;
    std::size_t capacity_ = 0;
};

OpenCLState& GetState(int platform) {
    static std::mutex mutex;
    static std::map<int, std::unique_ptr<OpenCLState>> states;

    std::lock_guard<std::mutex> lock(mutex);

    std::unique_ptr<OpenCLState>& state = states[platform];
    if (!state) {
        auto created = std::make_unique<OpenCLState>();
        created->Init(platform);
        state = std::move(created);
    }
    return *state;
}

}  // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    const std::size_t n = input.size();
    if (n == 0) {
        return {};
    }

    OpenCLState& state = GetState(platform);

    state.Submit(input.data(), n);

    // The zero-filled allocation runs on the CPU while the input transfer
    // and the kernel are in flight on the GPU.
    std::vector<float> result(n);

    state.Await(result.data(), n);
    return result;
}
