#include "gelu_ocl.h"

#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <cstdio>
#include <future>

namespace {

const char* kKernelSource = R"CLC(
//------------------------------------------------------------------
//function: gelu
//purpose: считает GELU через exp вместо tanh, каждый поток обрабатывает 4 соседних элемента
//------------------------------------------------------------------
__kernel void gelu(__global const float* in, __global float* out,
                   const uint size) {
    const float k0 = -2.0f * 0.7978845608028654f;
    const float k1 = k0 * 0.044715f;
    const uint i = get_global_id(0);
    const uint base = i * 4;
    if (base + 3 < size) {
        const float4 x = vload4(i, in);
        vstore4(x / (1.0f + native_exp(x * (k0 + k1 * x * x))), i, out);
    } else {
        for (uint j = base; j < size; ++j) {
            const float x = in[j];
            out[j] = x / (1.0f + native_exp(x * (k0 + k1 * x * x)));
        }
    }
}
)CLC";

struct OclState {
    int platform = -1;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_mem in = nullptr;
    cl_mem out = nullptr;
    size_t capacity = 0;

    //------------------------------------------------------------------
    //function: Release
    //purpose: освобождает все объекты OpenCL и сбрасывает состояние, чтобы его можно было создать заново
    //------------------------------------------------------------------
    void Release() {
        if (in) clReleaseMemObject(in);
        if (out) clReleaseMemObject(out);
        if (kernel) clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
        *this = OclState();
    }
};

OclState g_state;

//------------------------------------------------------------------
//function: Init
//purpose: один раз выбирает GPU на заданной платформе, создаёт контекст, очередь и компилирует ядро
//------------------------------------------------------------------
bool Init(int platform) {
    if (g_state.platform == platform && g_state.kernel) return true;
    g_state.Release();

    cl_uint platform_count = 0;
    clGetPlatformIDs(0, nullptr, &platform_count);
    if (platform < 0 || static_cast<cl_uint>(platform) >= platform_count) {
        std::fprintf(stderr, "OpenCL platform %d not found\n", platform);
        return false;
    }
    std::vector<cl_platform_id> platforms(platform_count);
    clGetPlatformIDs(platform_count, platforms.data(), nullptr);

    cl_device_id device = nullptr;
    if (clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 1, &device,
                       nullptr) != CL_SUCCESS) {
        std::fprintf(stderr, "No GPU device on OpenCL platform %d\n", platform);
        return false;
    }

    cl_int err = CL_SUCCESS;
    g_state.context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    g_state.queue = clCreateCommandQueue(g_state.context, device, 0, &err);
    g_state.program = clCreateProgramWithSource(g_state.context, 1,
                                                &kKernelSource, nullptr, &err);
    err = clBuildProgram(g_state.program, 1, &device, "-cl-fast-relaxed-math",
                         nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(g_state.program, device, CL_PROGRAM_BUILD_LOG, 0,
                              nullptr, &log_size);
        std::vector<char> log(log_size + 1);
        clGetProgramBuildInfo(g_state.program, device, CL_PROGRAM_BUILD_LOG,
                              log_size, log.data(), nullptr);
        std::fprintf(stderr, "OpenCL build failed:\n%s\n", log.data());
        g_state.Release();
        return false;
    }
    g_state.kernel = clCreateKernel(g_state.program, "gelu", &err);
    g_state.platform = platform;
    return true;
}

//------------------------------------------------------------------
//function: EnsureCapacity
//purpose: создаёт буферы на GPU только при росте размера, чтобы не выделять память на каждом вызове
//------------------------------------------------------------------
void EnsureCapacity(size_t size) {
    if (size <= g_state.capacity) return;
    if (g_state.in) clReleaseMemObject(g_state.in);
    if (g_state.out) clReleaseMemObject(g_state.out);
    cl_int err = CL_SUCCESS;
    g_state.in = clCreateBuffer(g_state.context, CL_MEM_READ_ONLY,
                                size * sizeof(float), nullptr, &err);
    g_state.out = clCreateBuffer(g_state.context, CL_MEM_WRITE_ONLY,
                                 size * sizeof(float), nullptr, &err);
    g_state.capacity = size;
}

}

//------------------------------------------------------------------
//function: GeluOCL
//purpose: считает GELU для вектора на GPU через OpenCL, пока CPU параллельно выделяет память под результат
//------------------------------------------------------------------
std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    const size_t size = input.size();
    if (size == 0 || !Init(platform)) return {};
    EnsureCapacity(size);
    auto result = std::async(std::launch::async,
                             [size] { return std::vector<float>(size); });

    const size_t bytes = size * sizeof(float);
    clEnqueueWriteBuffer(g_state.queue, g_state.in, CL_FALSE, 0, bytes,
                         input.data(), 0, nullptr, nullptr);

    const cl_uint count = static_cast<cl_uint>(size);
    clSetKernelArg(g_state.kernel, 0, sizeof(cl_mem), &g_state.in);
    clSetKernelArg(g_state.kernel, 1, sizeof(cl_mem), &g_state.out);
    clSetKernelArg(g_state.kernel, 2, sizeof(cl_uint), &count);

    const size_t local = 256;
    const size_t work = (size + 3) / 4;
    const size_t global = (work + local - 1) / local * local;
    clEnqueueNDRangeKernel(g_state.queue, g_state.kernel, 1, nullptr, &global,
                           &local, 0, nullptr, nullptr);
    clFlush(g_state.queue);

    std::vector<float> output = result.get();
    clEnqueueReadBuffer(g_state.queue, g_state.out, CL_TRUE, 0, bytes,
                        output.data(), 0, nullptr, nullptr);
    return output;
}
