#include "gelu_ocl.h"

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <cstddef>
#include <vector>

namespace {

const char* kGeluKernelSource = R"(
inline float gelu_fast(float x) {
  const float kMinusTwoSqrt2OverPi = -1.5957691216057307f;
  const float kCubicCoeff = -0.07135481627260025f;
  float arg = x * (kMinusTwoSqrt2OverPi + kCubicCoeff * x * x);
  return x / (1.0f + native_exp(arg));
}

__kernel void gelu_kernel_vec4(__global float* data, ulong n) {
  ulong idx = (ulong)get_global_id(0);
  ulong vec_n = n / 4;

  if (idx < vec_n) {
    __global float4* vec_data = (__global float4*)data;
    float4 v = vec_data[idx];
    v.x = gelu_fast(v.x);
    v.y = gelu_fast(v.y);
    v.z = gelu_fast(v.z);
    v.w = gelu_fast(v.w);
    vec_data[idx] = v;
  }

  if (idx == 0) {
    for (ulong i = vec_n * 4; i < n; ++i) {
      data[i] = gelu_fast(data[i]);
    }
  }
}
)";

struct OclWorkspace {
  int cached_platform = -1;
  cl_context context = nullptr;
  cl_command_queue queue = nullptr;
  cl_program program = nullptr;
  cl_kernel kernel = nullptr;
  cl_mem d_data = nullptr;
  std::size_t capacity_bytes = 0;

  ~OclWorkspace() {
    release_all();
  }

  void release_all() {
    if (d_data != nullptr) {
      clReleaseMemObject(d_data);
      d_data = nullptr;
    }
    capacity_bytes = 0;
    if (kernel != nullptr) {
      clReleaseKernel(kernel);
      kernel = nullptr;
    }
    if (program != nullptr) {
      clReleaseProgram(program);
      program = nullptr;
    }
    if (queue != nullptr) {
      clReleaseCommandQueue(queue);
      queue = nullptr;
    }
    if (context != nullptr) {
      clReleaseContext(context);
      context = nullptr;
    }
    cached_platform = -1;
  }

  void ensure_init(int platform_idx) {
    if (cached_platform == platform_idx && kernel != nullptr) {
      return;
    }

    release_all();

    cl_uint num_platforms = 0;
    clGetPlatformIDs(0, nullptr, &num_platforms);
    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    cl_platform_id platform_id = platforms[platform_idx];

    cl_device_id device_id = nullptr;
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, nullptr);

    context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, nullptr);
    queue = clCreateCommandQueue(context, device_id, 0, nullptr);

    program = clCreateProgramWithSource(context, 1, &kGeluKernelSource, nullptr, nullptr);
    clBuildProgram(program, 1, &device_id, "-cl-fast-relaxed-math", nullptr, nullptr);

    kernel = clCreateKernel(program, "gelu_kernel_vec4", nullptr);
    cached_platform = platform_idx;
  }

  void ensure_buffer(std::size_t bytes) {
    if (bytes > capacity_bytes) {
      if (d_data != nullptr) {
        clReleaseMemObject(d_data);
      }
      d_data = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, nullptr, nullptr);
      capacity_bytes = bytes;
    }
  }
};

} // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
  const std::size_t size = input.size();
  if (size == 0) {
    return {};
  }

  static OclWorkspace ws;
  ws.ensure_init(platform);

  const std::size_t bytes = size * sizeof(float);
  ws.ensure_buffer(bytes);

  clEnqueueWriteBuffer(ws.queue, ws.d_data, CL_FALSE, 0, bytes, input.data(), 0, nullptr, nullptr);

  const cl_ulong n_arg = static_cast<cl_ulong>(size);
  clSetKernelArg(ws.kernel, 0, sizeof(cl_mem), &ws.d_data);
  clSetKernelArg(ws.kernel, 1, sizeof(cl_ulong), &n_arg);

  const std::size_t local_size = 256;
  const std::size_t vec_n = size / 4;
  const std::size_t work_items = (vec_n > 0) ? vec_n : 1;
  const std::size_t global_size = ((work_items + local_size - 1) / local_size) * local_size;

  clEnqueueNDRangeKernel(ws.queue, ws.kernel, 1, nullptr, &global_size, &local_size, 0, nullptr, nullptr);
  clFlush(ws.queue);

  std::vector<float> output(size);

  clEnqueueReadBuffer(ws.queue, ws.d_data, CL_TRUE, 0, bytes, output.data(), 0, nullptr, nullptr);

  return output;
}
