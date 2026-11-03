import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

kernel_code = """
#define BLOCK_SIZE 256

__global__ void layernorm(const float* __restrict__ input, 
                          const float* __restrict__ gamma, 
                          const float* __restrict__ beta, 
                          float* __restrict__ output, 
                          int cols, 
                          float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    __shared__ float s_reduce[BLOCK_SIZE];

    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        local_sum += row_in[i];
    }
    s_reduce[tid] = local_sum;
    __syncthreads();

    #pragma unroll
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_reduce[tid] += s_reduce[tid + stride];
        }
        __syncthreads();
    }
    const float mean = s_reduce[0] / cols;

    float local_var = 0.0f;
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        float diff = row_in[i] - mean;
        local_var += diff * diff;
    }
    s_reduce[tid] = local_var;
    __syncthreads();

    #pragma unroll
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_reduce[tid] += s_reduce[tid + stride];
        }
        __syncthreads();
    }
    
    const float var = s_reduce[0] / cols;
    const float inv_std = rsqrtf(var + eps);

    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        row_out[i] = gamma[i] * (row_in[i] - mean) * inv_std + beta[i];
    }
}
"""


class GPUWorkspace:
    def __init__(self):
        self.kernel = None
        self.stream = None
        self.d_in = None
        self.d_gamma = None
        self.d_beta = None
        self.d_out = None
        self.cap = 0
        self.row_cap = 0

    def compile_if_needed(self):
        if self.kernel is None:
            mod = SourceModule(kernel_code)
            self.kernel = mod.get_function("layernorm")
            self.stream = drv.Stream()

    def allocate_if_needed(self, bytes_input, bytes_row):
        if self.cap < bytes_input:
            if self.d_in: self.d_in.free()
            if self.d_out: self.d_out.free()
            self.d_in = drv.mem_alloc(bytes_input)
            self.d_out = drv.mem_alloc(bytes_input)
            self.cap = bytes_input

        if self.row_cap < bytes_row:
            if self.d_gamma: self.d_gamma.free()
            if self.d_beta: self.d_beta.free()
            self.d_gamma = drv.mem_alloc(bytes_row)
            self.d_beta = drv.mem_alloc(bytes_row)
            self.row_cap = bytes_row

_workspace = GPUWorkspace()


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    """
    Apply Layer Normalization to each row of the input matrix.

    Parameters
    ----------
    input : list or numpy.ndarray of float
        Flattened matrix in row‑major order. Its length must be divisible by row_size.
    gamma : list or numpy.ndarray of float
        Scale parameter, length = row_size.
    beta : list or numpy.ndarray of float
        Shift parameter, length = row_size.
    row_size : int
        Number of features per row (i.e., number of columns).
    eps : float, optional
        Small constant for numerical stability.

    Returns
    -------
    numpy.ndarray
        Flattened matrix of the same shape as input, containing the row‑wise
        normalized results.
    """
    input_np = np.asarray(input, dtype=np.float32)
    gamma_np = np.asarray(gamma, dtype=np.float32)
    beta_np = np.asarray(beta, dtype=np.float32)

    size = input_np.size
    if size == 0 or row_size == 0:
        return np.array([], dtype=np.float32)

    _workspace.compile_if_needed()
    _workspace.allocate_if_needed(input_np.nbytes, gamma_np.nbytes)

    drv.memcpy_htod_async(_workspace.d_in, input_np, _workspace.stream)
    drv.memcpy_htod_async(_workspace.d_gamma, gamma_np, _workspace.stream)
    drv.memcpy_htod_async(_workspace.d_beta, beta_np, _workspace.stream)

    block = (256, 1, 1)
    row_count = int(size // row_size)
    grid = (row_count, 1)
    
    _workspace.kernel(
        _workspace.d_in, _workspace.d_gamma, _workspace.d_beta, 
        _workspace.d_out, np.int32(row_size), np.float32(eps), 
        block=block, grid=grid, stream=_workspace.stream
    )

    result = np.empty_like(input_np)
    drv.memcpy_dtoh_async(result, _workspace.d_out, _workspace.stream)
    
    _workspace.stream.synchronize()

    return result
