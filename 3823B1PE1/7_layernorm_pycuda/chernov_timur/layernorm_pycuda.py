import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

KERNEL_CODE = """
#define BLOCK_SIZE 256

__global__ void layernorm_kernel(const float* __restrict__ input, 
                                 const float* __restrict__ gamma, 
                                 const float* __restrict__ beta, 
                                 float* __restrict__ output, 
                                 int row_size, 
                                 float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    const float* row_in = input + row * row_size;
    float* row_out = output + row * row_size;

    __shared__ float shared_mem[BLOCK_SIZE];

    float local_sum = 0.0f;
    for (int i = tid; i < row_size; i += BLOCK_SIZE) {
        local_sum += row_in[i];
    }
    shared_mem[tid] = local_sum;
    __syncthreads();

    #pragma unroll
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    const float mean = shared_mem[0] / row_size;

    float local_var = 0.0f;
    for (int i = tid; i < row_size; i += BLOCK_SIZE) {
        float diff = row_in[i] - mean;
        local_var += diff * diff;
    }
    shared_mem[tid] = local_var;
    __syncthreads();

    #pragma unroll
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    float variance = shared_mem[0] / row_size;

    if (variance < 0.0f) {
        variance = 0.0f;
    }
    
    const float inv_std = rsqrtf(variance + eps);

    for (int i = tid; i < row_size; i += BLOCK_SIZE) {
        row_out[i] = gamma[i] * (row_in[i] - mean) * inv_std + beta[i];
    }
}
"""

class _GPUWorkspace:
    def __init__(self):
        self.kernel = None
        self.stream = None
        self.d_in = None
        self.d_gamma = None
        self.d_beta = None
        self.d_out = None
        self.cap_bytes = 0
        self.row_cap_bytes = 0

    def compile_if_needed(self):
        if self.kernel is None:
            mod = SourceModule(KERNEL_CODE)
            self.kernel = mod.get_function("layernorm_kernel")
            self.stream = drv.Stream()

    def allocate_if_needed(self, total_bytes, row_bytes):
        if self.cap_bytes < total_bytes:
            if self.d_in: self.d_in.free()
            if self.d_out: self.d_out.free()
            self.d_in = drv.mem_alloc(total_bytes)
            self.d_out = drv.mem_alloc(total_bytes)
            self.cap_bytes = total_bytes

        if self.row_cap_bytes < row_bytes:
            if self.d_gamma: self.d_gamma.free()
            if self.d_beta: self.d_beta.free()
            self.d_gamma = drv.mem_alloc(row_bytes)
            self.d_beta = drv.mem_alloc(row_bytes)
            self.row_cap_bytes = row_bytes

_workspace = _GPUWorkspace()

def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    input_arr = np.asarray(input, dtype=np.float32)
    orig_shape = input_arr.shape
    input_np = np.ascontiguousarray(input_arr).ravel()
    gamma_np = np.ascontiguousarray(gamma, dtype=np.float32).ravel()
    beta_np = np.ascontiguousarray(beta, dtype=np.float32).ravel()

    total_size = input_np.size
    if total_size == 0 or row_size == 0:
        return np.empty(orig_shape, dtype=np.float32)

    _workspace.compile_if_needed()
    _workspace.allocate_if_needed(input_np.nbytes, gamma_np.nbytes)

    drv.memcpy_htod_async(_workspace.d_in, input_np, _workspace.stream)
    drv.memcpy_htod_async(_workspace.d_gamma, gamma_np, _workspace.stream)
    drv.memcpy_htod_async(_workspace.d_beta, beta_np, _workspace.stream)

    block = (256, 1, 1)
    row_count = int(total_size // row_size)
    grid = (row_count, 1)
    
    _workspace.kernel(
        _workspace.d_in, _workspace.d_gamma, _workspace.d_beta, 
        _workspace.d_out, np.int32(row_size), np.float32(eps), 
        block=block, grid=grid, stream=_workspace.stream
    )

    result = np.empty_like(input_np)
    drv.memcpy_dtoh_async(result, _workspace.d_out, _workspace.stream)
    
    _workspace.stream.synchronize()

    return result.reshape(orig_shape)