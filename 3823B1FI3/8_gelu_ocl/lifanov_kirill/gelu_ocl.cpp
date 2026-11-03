#include "gelu_ocl.h"

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#include <CL/cl.h>

#include <cmath>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

    constexpr char kSource[] = R"CLC(
__kernel void gelu_vec4(__global float4* v, uint n4) {
    uint i = get_global_id(0);
    if (i < n4) {
        float4 x = v[i];
        float4 a = (float4)(1.5957691216f) * x * fma(x * x, (float4)(0.044715f), (float4)(1.0f));
        float4 d = native_exp2(-fabs(a) * 1.4426950409f);
        float4 r = (float4)(1.0f) / ((float4)(1.0f) + d);
        v[i] = x * select(d * r, r, a >= 0.0f);
    }
}
)CLC";

    void Check(cl_int err, const char* what) {
        if (err != CL_SUCCESS)
            throw std::runtime_error(std::string("OpenCL: ") + what + " failed, code " + std::to_string(err));
    }

    struct Ctx {
        cl_context context = nullptr;
        cl_command_queue queue = nullptr;
        cl_program program = nullptr;
        cl_kernel kernel = nullptr;
        cl_mem buf = nullptr;
        size_t capacity = 0;
        size_t local = 256;

        ~Ctx() {
            if (buf) clReleaseMemObject(buf);
            if (kernel) clReleaseKernel(kernel);
            if (program) clReleaseProgram(program);
            if (queue) clReleaseCommandQueue(queue);
            if (context) clReleaseContext(context);
        }

        void Init(int platform) {
            cl_uint np = 0;
            Check(clGetPlatformIDs(0, nullptr, &np), "clGetPlatformIDs");
            if (platform < 0 || static_cast<cl_uint>(platform) >= np)
                throw std::runtime_error("OpenCL: platform index out of range");
            std::vector<cl_platform_id> platforms(np);
            Check(clGetPlatformIDs(np, platforms.data(), nullptr), "clGetPlatformIDs");

            cl_device_id device = nullptr;
            Check(clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 1, &device, nullptr),
                "clGetDeviceIDs");

            cl_int err = CL_SUCCESS;
            context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
            Check(err, "clCreateContext");
            queue = clCreateCommandQueue(context, device, 0, &err);
            Check(err, "clCreateCommandQueue");

            const char* src = kSource;
            program = clCreateProgramWithSource(context, 1, &src, nullptr, &err);
            Check(err, "clCreateProgramWithSource");

            err = clBuildProgram(program, 1, &device, "-cl-std=CL1.2", nullptr, nullptr);
            if (err != CL_SUCCESS) {
                size_t sz = 0;
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &sz);
                std::string log(sz, '\0');
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sz, &log[0], nullptr);
                throw std::runtime_error("OpenCL build failed:\n" + log);
            }

            kernel = clCreateKernel(program, "gelu_vec4", &err);
            Check(err, "clCreateKernel");

            size_t wg = 0;
            clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(wg), &wg, nullptr);
            if (wg > 0 && wg < local) local = wg;
        }

        void Reserve(size_t bytes) {
            if (bytes <= capacity) return;
            if (buf) { clReleaseMemObject(buf); buf = nullptr; capacity = 0; }
            cl_int err = CL_SUCCESS;
            buf = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, nullptr, &err);
            Check(err, "clCreateBuffer");
            capacity = bytes;
        }
    };

    std::mutex g_mutex;
    std::unordered_map<int, std::unique_ptr<Ctx>> g_cache;

    Ctx* GetOrCreate(int platform) {
        auto it = g_cache.find(platform);
        if (it != g_cache.end()) return it->second.get();
        std::unique_ptr<Ctx> c(new Ctx());
        c->Init(platform);
        Ctx* raw = c.get();
        g_cache[platform] = std::move(c);
        return raw;
    }

    inline float GeluCpu(float x) {
        float a = 1.5957691216f * x * (1.0f + 0.044715f * x * x);
        return x / (1.0f + std::exp(-a));
    }

}  // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    const size_t n = input.size();
    if (n == 0) return {};

    std::lock_guard<std::mutex> lock(g_mutex);
    Ctx* c = GetOrCreate(platform);

    const size_t n4 = n / 4;
    const size_t main_n = n4 * 4;
    const size_t bytes = main_n * sizeof(float);

    std::vector<float> output;

    if (main_n > 0) {
        c->Reserve(bytes);

        Check(clEnqueueWriteBuffer(c->queue, c->buf, CL_FALSE, 0, bytes, input.data(),
            0, nullptr, nullptr), "write");

        cl_uint n4_arg = static_cast<cl_uint>(n4);
        Check(clSetKernelArg(c->kernel, 0, sizeof(cl_mem), &c->buf), "arg0");
        Check(clSetKernelArg(c->kernel, 1, sizeof(cl_uint), &n4_arg), "arg1");

        const size_t global = (n4 + c->local - 1) / c->local * c->local;
        Check(clEnqueueNDRangeKernel(c->queue, c->kernel, 1, nullptr, &global, &c->local,
            0, nullptr, nullptr), "kernel");
        clFlush(c->queue);

        output.resize(n);

        Check(clEnqueueReadBuffer(c->queue, c->buf, CL_FALSE, 0, bytes, output.data(),
            0, nullptr, nullptr), "read");
    }
    else {
        output.resize(n);
    }

    for (size_t i = main_n; i < n; ++i) output[i] = GeluCpu(input[i]);

    if (main_n > 0) Check(clFinish(c->queue), "clFinish");
    return output;
}