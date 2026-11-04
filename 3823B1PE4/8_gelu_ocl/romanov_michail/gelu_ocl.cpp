#include "gelu_ocl.h"

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <cstddef>
#include <vector>

namespace {

const char* kKernelSource = R"CLC(
__kernel void gelu_kernel(__global const float* in, __global float* out, int size) {
    const int idx = get_global_id(0);
    if (idx >= size) {
        return;
    }
    const float val = in[idx];
    const float val_sq = val * val;
    const float inner = 0.7978845608028654f * (val + 0.044715f * val_sq * val);
    out[idx] = val / (1.0f + exp(-2.0f * inner));
}

__kernel void gelu4_kernel(__global const float4* in, __global float4* out, int size4) {
    const int idx = get_global_id(0);
    if (idx >= size4) {
        return;
    }
    const float4 val = in[idx];
    const float4 val_sq = val * val;
    const float4 inner = 0.7978845608028654f * (val + 0.044715f * val_sq * val);
    out[idx] = val / (1.0f + exp(-2.0f * inner));
}
)CLC";

struct ContextState {
    int platform_idx = -1;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_kernel kernel4 = nullptr;
    cl_mem buffer_in = nullptr;
    cl_mem buffer_out = nullptr;
    std::size_t buffer_capacity = 0;
};

ContextState& GetState() {
    static ContextState state;
    return state;
}

void FreeBuffers(ContextState& state) {
    if (state.buffer_in != nullptr) {
        clReleaseMemObject(state.buffer_in);
        state.buffer_in = nullptr;
    }
    if (state.buffer_out != nullptr) {
        clReleaseMemObject(state.buffer_out);
        state.buffer_out = nullptr;
    }
    state.buffer_capacity = 0;
}

void CleanState(ContextState& state) {
    FreeBuffers(state);
    if (state.kernel != nullptr) {
        clReleaseKernel(state.kernel);
        state.kernel = nullptr;
    }
    if (state.kernel4 != nullptr) {
        clReleaseKernel(state.kernel4);
        state.kernel4 = nullptr;
    }
    if (state.program != nullptr) {
        clReleaseProgram(state.program);
        state.program = nullptr;
    }
    if (state.queue != nullptr) {
        clReleaseCommandQueue(state.queue);
        state.queue = nullptr;
    }
    if (state.context != nullptr) {
        clReleaseContext(state.context);
        state.context = nullptr;
    }
    state.platform_idx = -1;
}

bool InitializeOpenCL(int platform_index) {
    ContextState& state = GetState();
    if (state.platform_idx == platform_index && state.context != nullptr && 
        state.queue != nullptr && state.kernel != nullptr && state.kernel4 != nullptr) {
        return true;
    }
    CleanState(state);

    cl_uint num_platforms = 0;
    if (clGetPlatformIDs(0, nullptr, &num_platforms) != CL_SUCCESS || num_platforms == 0) {
        return false;
    }
    if (platform_index < 0 || static_cast<cl_uint>(platform_index) >= num_platforms) {
        return false;
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    if (clGetPlatformIDs(num_platforms, platforms.data(), nullptr) != CL_SUCCESS) {
        return false;
    }
    cl_platform_id platform = platforms[static_cast<std::size_t>(platform_index)];

    cl_device_id device = nullptr;
    cl_uint num_devices = 0;
    cl_int err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
    if (err == CL_SUCCESS && num_devices > 0) {
        std::vector<cl_device_id> devices(num_devices);
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr) != CL_SUCCESS) {
            return false;
        }
        device = devices[0];
    } else {
        return false;
    }

    cl_int status = CL_SUCCESS;
    cl_context_properties props[] = {CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform), 0};
    state.context = clCreateContext(props, 1, &device, nullptr, nullptr, &status);
    if (status != CL_SUCCESS || state.context == nullptr) {
        CleanState(state);
        return false;
    }

    state.queue = clCreateCommandQueue(state.context, device, 0, &status);
    if (status != CL_SUCCESS || state.queue == nullptr) {
        CleanState(state);
        return false;
    }

    state.program = clCreateProgramWithSource(state.context, 1, &kKernelSource, nullptr, &status);
    if (status != CL_SUCCESS || state.program == nullptr) {
        CleanState(state);
        return false;
    }

    if (clBuildProgram(state.program, 1, &device, nullptr, nullptr, nullptr) != CL_SUCCESS) {
        CleanState(state);
        return false;
    }

    state.kernel = clCreateKernel(state.program, "gelu_kernel", &status);
    if (status != CL_SUCCESS || state.kernel == nullptr) {
        CleanState(state);
        return false;
    }

    state.kernel4 = clCreateKernel(state.program, "gelu4_kernel", &status);
    if (status != CL_SUCCESS || state.kernel4 == nullptr) {
        CleanState(state);
        return false;
    }

    state.platform_idx = platform_index;
    return true;
}

bool EnsureMemoryBuffers(std::size_t count) {
    ContextState& state = GetState();
    if (state.buffer_capacity >= count && state.buffer_in != nullptr && state.buffer_out != nullptr) {
        return true;
    }
    FreeBuffers(state);

    cl_int status = CL_SUCCESS;
    std::size_t size_bytes = count * sizeof(float);
    state.buffer_in = clCreateBuffer(state.context, CL_MEM_READ_ONLY, size_bytes, nullptr, &status);
    if (status != CL_SUCCESS || state.buffer_in == nullptr) {
        FreeBuffers(state);
        return false;
    }

    state.buffer_out = clCreateBuffer(state.context, CL_MEM_WRITE_ONLY, size_bytes, nullptr, &status);
    if (status != CL_SUCCESS || state.buffer_out == nullptr) {
        FreeBuffers(state);
        return false;
    }

    state.buffer_capacity = count;
    return true;
}

}

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    std::size_t n = input.size();
    if (n == 0) {
        return {};
    }

    if (!InitializeOpenCL(platform) || !EnsureMemoryBuffers(n)) {
        return {};
    }

    ContextState& state = GetState();
    std::size_t bytes = n * sizeof(float);

    if (clEnqueueWriteBuffer(state.queue, state.buffer_in, CL_FALSE, 0, bytes, input.data(), 0, nullptr, nullptr) != CL_SUCCESS) {
        return {};
    }

    std::vector<float> output(n);
    constexpr std::size_t local_work_size = 256;
    cl_int status = CL_SUCCESS;

    if ((n & 3u) == 0) {
        int n4 = static_cast<int>(n >> 2);
        status = clSetKernelArg(state.kernel4, 0, sizeof(cl_mem), &state.buffer_in);
        status |= clSetKernelArg(state.kernel4, 1, sizeof(cl_mem), &state.buffer_out);
        status |= clSetKernelArg(state.kernel4, 2, sizeof(int), &n4);
        if (status != CL_SUCCESS) {
            return {};
        }

        std::size_t global_work_size = (static_cast<std::size_t>(n4) + local_work_size - 1) / local_work_size * local_work_size;
        if (clEnqueueNDRangeKernel(state.queue, state.kernel4, 1, nullptr, &global_work_size, &local_work_size, 0, nullptr, nullptr) != CL_SUCCESS) {
            return {};
        }
    } else {
        int nn = static_cast<int>(n);
        status = clSetKernelArg(state.kernel, 0, sizeof(cl_mem), &state.buffer_in);
        status |= clSetKernelArg(state.kernel, 1, sizeof(cl_mem), &state.buffer_out);
        status |= clSetKernelArg(state.kernel, 2, sizeof(int), &nn);
        if (status != CL_SUCCESS) {
            return {};
        }

        std::size_t global_work_size = (n + local_work_size - 1) / local_work_size * local_work_size;
        if (clEnqueueNDRangeKernel(state.queue, state.kernel, 1, nullptr, &global_work_size, &local_work_size, 0, nullptr, nullptr) != CL_SUCCESS) {
            return {};
        }
    }

    if (clEnqueueReadBuffer(state.queue, state.buffer_out, CL_TRUE, 0, bytes, output.data(), 0, nullptr, nullptr) != CL_SUCCESS) {
        return {};
    }

    return output;
}