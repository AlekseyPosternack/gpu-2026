#include "gelu_ocl.h"

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <cstddef>
#include <vector>

namespace {

const char* kGeluSource = R"CLC(
__kernel void gelu(__global const float* in, __global float* out, int n) {
    const int i = get_global_id(0);
    if (i >= n) {
        return;
    }
    const float x = in[i];
    const float x2 = x * x;
    const float s = 0.7978845608028654f * (x + 0.044715f * x2 * x);
    out[i] = x / (1.0f + exp(-2.0f * s));
}

__kernel void gelu4(__global const float4* in, __global float4* out, int n4) {
    const int i = get_global_id(0);
    if (i >= n4) {
        return;
    }
    const float4 x = in[i];
    const float4 x2 = x * x;
    const float4 s = 0.7978845608028654f * (x + 0.044715f * x2 * x);
    out[i] = x / (1.0f + exp(-2.0f * s));
}
)CLC";

struct Runtime {
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

Runtime& GetRuntime() {
    static Runtime rt;
    return rt;
}

void ReleaseBuffers(Runtime& rt) {
    if (rt.d_in != nullptr) {
        clReleaseMemObject(rt.d_in);
        rt.d_in = nullptr;
    }
    if (rt.d_out != nullptr) {
        clReleaseMemObject(rt.d_out);
        rt.d_out = nullptr;
    }
    rt.cap = 0;
}

void ReleaseAll(Runtime& rt) {
    ReleaseBuffers(rt);
    if (rt.kernel != nullptr) {
        clReleaseKernel(rt.kernel);
        rt.kernel = nullptr;
    }
    if (rt.kernel4 != nullptr) {
        clReleaseKernel(rt.kernel4);
        rt.kernel4 = nullptr;
    }
    if (rt.program != nullptr) {
        clReleaseProgram(rt.program);
        rt.program = nullptr;
    }
    if (rt.queue != nullptr) {
        clReleaseCommandQueue(rt.queue);
        rt.queue = nullptr;
    }
    if (rt.ctx != nullptr) {
        clReleaseContext(rt.ctx);
        rt.ctx = nullptr;
    }
    rt.platform = -1;
}

bool Init(int platform_index) {
    Runtime& rt = GetRuntime();
    if (rt.platform == platform_index && rt.ctx != nullptr && rt.queue != nullptr &&
        rt.kernel != nullptr && rt.kernel4 != nullptr) {
        return true;
    }
    ReleaseAll(rt);

    cl_uint nplatforms = 0;
    if (clGetPlatformIDs(0, nullptr, &nplatforms) != CL_SUCCESS || nplatforms == 0) {
        return false;
    }
    if (platform_index < 0 || static_cast<cl_uint>(platform_index) >= nplatforms) {
        return false;
    }

    std::vector<cl_platform_id> platforms(nplatforms);
    if (clGetPlatformIDs(nplatforms, platforms.data(), nullptr) != CL_SUCCESS) {
        return false;
    }
    const cl_platform_id plat = platforms[static_cast<std::size_t>(platform_index)];

    cl_device_id device = nullptr;
    cl_uint ndev = 0;
    cl_int derr = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &ndev);
    if (derr == CL_SUCCESS && ndev > 0) {
        std::vector<cl_device_id> devices(ndev);
        if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, ndev, devices.data(), nullptr) != CL_SUCCESS) {
            return false;
        }
        device = devices[0];
    } else if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &device, &ndev) != CL_SUCCESS ||
               device == nullptr) {
        return false;
    }

    cl_int err = CL_SUCCESS;
    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(plat), 0};
    rt.ctx = clCreateContext(props, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS || rt.ctx == nullptr) {
        ReleaseAll(rt);
        return false;
    }
    rt.queue = clCreateCommandQueue(rt.ctx, device, 0, &err);
    if (err != CL_SUCCESS || rt.queue == nullptr) {
        ReleaseAll(rt);
        return false;
    }
    rt.program = clCreateProgramWithSource(rt.ctx, 1, &kGeluSource, nullptr, &err);
    if (err != CL_SUCCESS || rt.program == nullptr) {
        ReleaseAll(rt);
        return false;
    }
    if (clBuildProgram(rt.program, 1, &device, nullptr, nullptr, nullptr) != CL_SUCCESS) {
        ReleaseAll(rt);
        return false;
    }
    rt.kernel = clCreateKernel(rt.program, "gelu", &err);
    if (err != CL_SUCCESS || rt.kernel == nullptr) {
        ReleaseAll(rt);
        return false;
    }
    rt.kernel4 = clCreateKernel(rt.program, "gelu4", &err);
    if (err != CL_SUCCESS || rt.kernel4 == nullptr) {
        ReleaseAll(rt);
        return false;
    }
    rt.platform = platform_index;
    return true;
}

bool EnsureBuffers(std::size_t n) {
    Runtime& rt = GetRuntime();
    if (rt.cap >= n && rt.d_in != nullptr && rt.d_out != nullptr) {
        return true;
    }
    ReleaseBuffers(rt);
    cl_int err = CL_SUCCESS;
    const std::size_t bytes = n * sizeof(float);
    rt.d_in = clCreateBuffer(rt.ctx, CL_MEM_READ_ONLY, bytes, nullptr, &err);
    if (err != CL_SUCCESS || rt.d_in == nullptr) {
        ReleaseBuffers(rt);
        return false;
    }
    rt.d_out = clCreateBuffer(rt.ctx, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
    if (err != CL_SUCCESS || rt.d_out == nullptr) {
        ReleaseBuffers(rt);
        return false;
    }
    rt.cap = n;
    return true;
}

}

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    const std::size_t n = input.size();
    if (n == 0) {
        return {};
    }
    if (!Init(platform) || !EnsureBuffers(n)) {
        return {};
    }

    Runtime& rt = GetRuntime();
    const std::size_t bytes = n * sizeof(float);
    if (clEnqueueWriteBuffer(rt.queue, rt.d_in, CL_FALSE, 0, bytes, input.data(), 0, nullptr,
                             nullptr) != CL_SUCCESS) {
        return {};
    }

    std::vector<float> output(n);

    constexpr std::size_t kLocal = 256;
    cl_int err = CL_SUCCESS;
    if ((n & 3u) == 0) {
        const int n4 = static_cast<int>(n >> 2);
        err = clSetKernelArg(rt.kernel4, 0, sizeof(cl_mem), &rt.d_in);
        err |= clSetKernelArg(rt.kernel4, 1, sizeof(cl_mem), &rt.d_out);
        err |= clSetKernelArg(rt.kernel4, 2, sizeof(int), &n4);
        if (err != CL_SUCCESS) {
            return {};
        }
        const std::size_t global =
            (static_cast<std::size_t>(n4) + kLocal - 1) / kLocal * kLocal;
        if (clEnqueueNDRangeKernel(rt.queue, rt.kernel4, 1, nullptr, &global, &kLocal, 0, nullptr,
                                   nullptr) != CL_SUCCESS) {
            return {};
        }
    } else {
        const int nn = static_cast<int>(n);
        err = clSetKernelArg(rt.kernel, 0, sizeof(cl_mem), &rt.d_in);
        err |= clSetKernelArg(rt.kernel, 1, sizeof(cl_mem), &rt.d_out);
        err |= clSetKernelArg(rt.kernel, 2, sizeof(int), &nn);
        if (err != CL_SUCCESS) {
            return {};
        }
        const std::size_t global = (n + kLocal - 1) / kLocal * kLocal;
        if (clEnqueueNDRangeKernel(rt.queue, rt.kernel, 1, nullptr, &global, &kLocal, 0, nullptr,
                                   nullptr) != CL_SUCCESS) {
            return {};
        }
    }

    if (clEnqueueReadBuffer(rt.queue, rt.d_out, CL_TRUE, 0, bytes, output.data(), 0, nullptr,
                            nullptr) != CL_SUCCESS) {
        return {};
    }
    return output;
}
