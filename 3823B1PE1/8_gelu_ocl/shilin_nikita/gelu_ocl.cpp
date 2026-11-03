#include "gelu_ocl.h"

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <cstddef>

namespace {

const char* kGeluKernel = R"CLC(
__kernel void gelu_f4(__global const float4* in, __global float4* out, int n4) {
    const int i = get_global_id(0);
    if (i >= n4) {
        return;
    }
    const float4 x = in[i];
    const float4 x2 = x * x;
    const float4 s = 0.7978845608028654f * (x + 0.044715f * x2 * x);
    out[i] = x / (1.0f + native_exp(-2.0f * s));
}

__kernel void gelu(__global const float* in, __global float* out, int n) {
    const int i = get_global_id(0);
    if (i >= n) {
        return;
    }
    const float x = in[i];
    const float x2 = x * x;
    const float s = 0.7978845608028654f * (x + 0.044715f * x2 * x);
    out[i] = x / (1.0f + native_exp(-2.0f * s));
}
)CLC";

struct OclCache {
    int platform = -1;
    cl_context ctx = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_kernel kernel4 = nullptr;
    cl_mem d_in = nullptr;
    cl_mem d_out = nullptr;
    std::size_t cap = 0;
};

OclCache& Cache() {
    static OclCache cache;
    return cache;
}

void ReleaseBuffers(OclCache& cache) {
    if (cache.d_in != nullptr) {
        clReleaseMemObject(cache.d_in);
        cache.d_in = nullptr;
    }
    if (cache.d_out != nullptr) {
        clReleaseMemObject(cache.d_out);
        cache.d_out = nullptr;
    }
    cache.cap = 0;
}

void ReleaseAll(OclCache& cache) {
    ReleaseBuffers(cache);
    if (cache.kernel != nullptr) {
        clReleaseKernel(cache.kernel);
        cache.kernel = nullptr;
    }
    if (cache.kernel4 != nullptr) {
        clReleaseKernel(cache.kernel4);
        cache.kernel4 = nullptr;
    }
    if (cache.program != nullptr) {
        clReleaseProgram(cache.program);
        cache.program = nullptr;
    }
    if (cache.queue != nullptr) {
        clReleaseCommandQueue(cache.queue);
        cache.queue = nullptr;
    }
    if (cache.ctx != nullptr) {
        clReleaseContext(cache.ctx);
        cache.ctx = nullptr;
    }
    cache.platform = -1;
}

bool EnsureContext(int platform_index) {
    OclCache& cache = Cache();
    if (cache.platform == platform_index && cache.ctx != nullptr) {
        return true;
    }
    ReleaseAll(cache);

    cl_uint num_platforms = 0;
    if (clGetPlatformIDs(0, nullptr, &num_platforms) != CL_SUCCESS || num_platforms == 0) {
        return false;
    }
    if (platform_index < 0 || static_cast<cl_uint>(platform_index) >= num_platforms) {
        return false;
    }

    cl_platform_id platforms[16];
    const cl_uint to_query = num_platforms < 16u ? num_platforms : 16u;
    clGetPlatformIDs(to_query, platforms, nullptr);
    if (static_cast<cl_uint>(platform_index) >= to_query) {
        return false;
    }

    cl_device_id device = nullptr;
    if (clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_GPU, 1, &device, nullptr) !=
        CL_SUCCESS) {
        return false;
    }

    cl_int err = CL_SUCCESS;
    cache.ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        return false;
    }
    cache.queue = clCreateCommandQueue(cache.ctx, device, 0, &err);
    if (err != CL_SUCCESS) {
        return false;
    }
    cache.program = clCreateProgramWithSource(cache.ctx, 1, &kGeluKernel, nullptr, &err);
    if (err != CL_SUCCESS) {
        return false;
    }
    if (clBuildProgram(cache.program, 1, &device, "-cl-fast-relaxed-math", nullptr, nullptr) !=
        CL_SUCCESS) {
        return false;
    }
    cache.kernel = clCreateKernel(cache.program, "gelu", &err);
    if (err != CL_SUCCESS) {
        return false;
    }
    cache.kernel4 = clCreateKernel(cache.program, "gelu_f4", &err);
    if (err != CL_SUCCESS) {
        return false;
    }
    cache.platform = platform_index;
    return true;
}

bool EnsureBuffers(std::size_t n) {
    OclCache& cache = Cache();
    if (cache.cap >= n) {
        return true;
    }
    ReleaseBuffers(cache);
    cl_int err = CL_SUCCESS;
    const std::size_t bytes = n * sizeof(float);
    cache.d_in = clCreateBuffer(cache.ctx, CL_MEM_READ_ONLY, bytes, nullptr, &err);
    if (err != CL_SUCCESS) {
        return false;
    }
    cache.d_out = clCreateBuffer(cache.ctx, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
    if (err != CL_SUCCESS) {
        return false;
    }
    cache.cap = n;
    return true;
}

}

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    const int n = static_cast<int>(input.size());
    if (n == 0) {
        return {};
    }
    if (!EnsureContext(platform) || !EnsureBuffers(static_cast<std::size_t>(n))) {
        return std::vector<float>(static_cast<std::size_t>(n), 0.0f);
    }

    OclCache& cache = Cache();
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    clEnqueueWriteBuffer(cache.queue, cache.d_in, CL_FALSE, 0, bytes, input.data(), 0, nullptr,
                         nullptr);

    std::vector<float> output(static_cast<std::size_t>(n));

    constexpr std::size_t kLocal = 256;
    if ((n & 3) == 0) {
        const int n4 = n >> 2;
        clSetKernelArg(cache.kernel4, 0, sizeof(cl_mem), &cache.d_in);
        clSetKernelArg(cache.kernel4, 1, sizeof(cl_mem), &cache.d_out);
        clSetKernelArg(cache.kernel4, 2, sizeof(int), &n4);
        const std::size_t global =
            (static_cast<std::size_t>(n4) + kLocal - 1) / kLocal * kLocal;
        clEnqueueNDRangeKernel(cache.queue, cache.kernel4, 1, nullptr, &global, &kLocal, 0, nullptr,
                               nullptr);
    } else {
        clSetKernelArg(cache.kernel, 0, sizeof(cl_mem), &cache.d_in);
        clSetKernelArg(cache.kernel, 1, sizeof(cl_mem), &cache.d_out);
        clSetKernelArg(cache.kernel, 2, sizeof(int), &n);
        const std::size_t global = (static_cast<std::size_t>(n) + kLocal - 1) / kLocal * kLocal;
        clEnqueueNDRangeKernel(cache.queue, cache.kernel, 1, nullptr, &global, &kLocal, 0, nullptr,
                               nullptr);
    }

    clEnqueueReadBuffer(cache.queue, cache.d_out, CL_TRUE, 0, bytes, output.data(), 0, nullptr,
                        nullptr);
    return output;
}
