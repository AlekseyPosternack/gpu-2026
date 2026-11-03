#include "gelu_ocl.h"

#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

#define OCL_CHECK(call)                                                   \
    do {                                                                  \
        cl_int err_ = (call);                                             \
        if (err_ != CL_SUCCESS) {                                         \
            std::fprintf(stderr, "OpenCL error %d at %s:%d\n",            \
                         static_cast<int>(err_), __FILE__, __LINE__);     \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

namespace {

    const char* kKernelSource = R"CLC(
__kernel void gelu(__global float* data, const int n) {
    const float c1 = 1.5957691216057308f;   // 2 * sqrt(2 / pi)
    const float c2 = 0.0713548162726009f;   // c1 * 0.044715

    const int i = get_global_id(0);
    if (i < n) {
        const float x = data[i];
        const float u = x * (c1 + c2 * x * x);
        data[i] = x / (1.0f + native_exp(-u));
    }
}
)CLC";

    constexpr size_t kLocalSize = 256;  
    class OclState {
    public:
        static OclState& Instance() {
            static OclState state;
            return state;
        }

        void Init(int platform_index) {
            if (initialized_ && platform_index == platform_index_) {
                return;
            }
            Release();

            
            cl_uint platform_count = 0;
            OCL_CHECK(clGetPlatformIDs(0, nullptr, &platform_count));
            if (platform_index < 0 || static_cast<cl_uint>(platform_index) >= platform_count) {
                std::fprintf(stderr, "OpenCL platform %d not found (%u available)\n",
                    platform_index, platform_count);
                std::exit(EXIT_FAILURE);
            }
            std::vector<cl_platform_id> platforms(platform_count);
            OCL_CHECK(clGetPlatformIDs(platform_count, platforms.data(), nullptr));

            
            cl_uint device_count = 0;
            const cl_int device_err = clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_GPU,
                0, nullptr, &device_count);
            if (device_err == CL_DEVICE_NOT_FOUND || device_count == 0) {
                std::fprintf(stderr, "OpenCL platform %d has no GPU devices\n", platform_index);
                std::exit(EXIT_FAILURE);
            }
            OCL_CHECK(device_err);
            std::vector<cl_device_id> devices(device_count);
            OCL_CHECK(clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_GPU,
                device_count, devices.data(), nullptr));
            device_ = devices[0];

            
            cl_int err = CL_SUCCESS;
            context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
            OCL_CHECK(err);
            queue_ = clCreateCommandQueue(context_, device_, 0, &err);
            OCL_CHECK(err);

            
            program_ = clCreateProgramWithSource(context_, 1, &kKernelSource, nullptr, &err);
            OCL_CHECK(err);
            err = clBuildProgram(program_, 1, &device_, "-cl-fast-relaxed-math", nullptr, nullptr);
            if (err != CL_SUCCESS) {
                PrintBuildLog();
                OCL_CHECK(err);
            }
            kernel_ = clCreateKernel(program_, "gelu", &err);
            OCL_CHECK(err);

            platform_index_ = platform_index;
            initialized_ = true;
        }

        
        cl_mem Buffer(size_t bytes) {
            if (bytes > buffer_bytes_) {
                if (buffer_ != nullptr) {
                    clReleaseMemObject(buffer_);
                }
                cl_int err = CL_SUCCESS;
                buffer_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, bytes, nullptr, &err);
                OCL_CHECK(err);
                buffer_bytes_ = bytes;
            }
            return buffer_;
        }

        cl_command_queue Queue() const { return queue_; }
        cl_kernel Kernel() const { return kernel_; }

        OclState(const OclState&) = delete;
        OclState& operator=(const OclState&) = delete;

    private:
        OclState() = default;
        ~OclState() { Release(); }

        void PrintBuildLog() {
            size_t size = 0;
            clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
            std::string log(size, '\0');
            clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, size, &log[0], nullptr);
            std::fprintf(stderr, "OpenCL build log:\n%s\n", log.c_str());
        }

        void Release() {
            if (buffer_ != nullptr) clReleaseMemObject(buffer_);
            if (kernel_ != nullptr) clReleaseKernel(kernel_);
            if (program_ != nullptr) clReleaseProgram(program_);
            if (queue_ != nullptr) clReleaseCommandQueue(queue_);
            if (context_ != nullptr) clReleaseContext(context_);
            buffer_ = nullptr;
            kernel_ = nullptr;
            program_ = nullptr;
            queue_ = nullptr;
            context_ = nullptr;
            buffer_bytes_ = 0;
            initialized_ = false;
        }

        bool initialized_ = false;
        int platform_index_ = -1;
        cl_device_id device_ = nullptr;
        cl_context context_ = nullptr;
        cl_command_queue queue_ = nullptr;
        cl_program program_ = nullptr;
        cl_kernel kernel_ = nullptr;
        cl_mem buffer_ = nullptr;
        size_t buffer_bytes_ = 0;
    };

}  // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    const int n = static_cast<int>(input.size());
    if (n == 0) {
        return {};
    }
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    OclState& ocl = OclState::Instance();
    ocl.Init(platform);
    cl_command_queue queue = ocl.Queue();
    cl_kernel kernel = ocl.Kernel();
    cl_mem buffer = ocl.Buffer(bytes);

   
    std::vector<float> output;
    std::thread alloc_thread([&output, n] { output.resize(n); });

    
    OCL_CHECK(clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, bytes, input.data(),
        0, nullptr, nullptr));

    OCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer));
    OCL_CHECK(clSetKernelArg(kernel, 1, sizeof(int), &n));
    
    const size_t global_size = (static_cast<size_t>(n) + kLocalSize - 1) / kLocalSize * kLocalSize;
    const size_t local_size = kLocalSize;
    OCL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, &local_size,
        0, nullptr, nullptr));

    alloc_thread.join();

    
    OCL_CHECK(clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, bytes, output.data(),
        0, nullptr, nullptr));

    return output;
}