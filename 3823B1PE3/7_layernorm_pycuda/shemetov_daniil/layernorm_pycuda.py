import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

_KERNEL_CODE = r"""
__global__ void layernorm_small_kernel(float* data,
                                       const float* __restrict__ gamma,
                                       const float* __restrict__ beta,
                                       int row_count,
                                       int row_size,
                                       float eps) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= row_count) {
    return;
  }

  float* row_ptr = data + (size_t)row * row_size;
  double sum = 0.0;
  double sum_sq = 0.0;

  for (int i = 0; i < row_size; ++i) {
    double v = (double)row_ptr[i];
    sum += v;
    sum_sq += v * v;
  }

  double inv_n = 1.0 / (double)row_size;
  double mean = sum * inv_n;
  double var = sum_sq * inv_n - mean * mean;
  if (var < 0.0) {
    var = 0.0;
  }

  float mean_f = (float)mean;
  float inv_std = rsqrtf((float)var + eps);

  for (int i = 0; i < row_size; ++i) {
    row_ptr[i] = (row_ptr[i] - mean_f) * inv_std * gamma[i] + beta[i];
  }
}

__global__ void layernorm_vec4_kernel(float* data,
                                      const float* __restrict__ gamma,
                                      const float* __restrict__ beta,
                                      int vec_row_size,
                                      float eps) {
  __shared__ double s_sum[256];
  __shared__ double s_sum_sq[256];

  int tid = threadIdx.x;
  int bdim = blockDim.x;
  size_t row = (size_t)blockIdx.x;

  float4* row_ptr = reinterpret_cast<float4*>(data) + row * vec_row_size;
  const float4* gamma4 = reinterpret_cast<const float4*>(gamma);
  const float4* beta4 = reinterpret_cast<const float4*>(beta);

  double local_sum = 0.0;
  double local_sum_sq = 0.0;

  for (int i = tid; i < vec_row_size; i += bdim) {
    float4 v = row_ptr[i];
    double vx = (double)v.x;
    double vy = (double)v.y;
    double vz = (double)v.z;
    double vw = (double)v.w;

    local_sum += vx + vy + vz + vw;
    local_sum_sq += vx * vx + vy * vy + vz * vz + vw * vw;
  }

  s_sum[tid] = local_sum;
  s_sum_sq[tid] = local_sum_sq;
  __syncthreads();

  for (int stride = bdim >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_sum[tid] += s_sum[tid + stride];
      s_sum_sq[tid] += s_sum_sq[tid + stride];
    }
    __syncthreads();
  }

  double inv_n = 1.0 / (double)(vec_row_size * 4);
  double mean = s_sum[0] * inv_n;
  double var = s_sum_sq[0] * inv_n - mean * mean;
  if (var < 0.0) {
    var = 0.0;
  }

  float mean_f = (float)mean;
  float inv_std = rsqrtf((float)var + eps);

  for (int i = tid; i < vec_row_size; i += bdim) {
    float4 v = row_ptr[i];
    float4 g = gamma4[i];
    float4 b = beta4[i];

    v.x = (v.x - mean_f) * inv_std * g.x + b.x;
    v.y = (v.y - mean_f) * inv_std * g.y + b.y;
    v.z = (v.z - mean_f) * inv_std * g.z + b.z;
    v.w = (v.w - mean_f) * inv_std * g.w + b.w;

    row_ptr[i] = v;
  }
}
"""


class _LayerNormWorkspace:
  def __init__(self):
    self.mod = None
    self.kernel_small = None
    self.kernel_vec4 = None
    self.stream = None
    self.d_data = None
    self.data_cap = 0
    self.d_gamma = None
    self.d_beta = None
    self.param_cap = 0

  def init_kernels(self):
    if self.mod is None:
      self.mod = SourceModule(_KERNEL_CODE)
      self.kernel_small = self.mod.get_function("layernorm_small_kernel")
      self.kernel_vec4 = self.mod.get_function("layernorm_vec4_kernel")
      self.stream = cuda.Stream()

  def ensure_buffers(self, total_elems, row_size):
    data_bytes = total_elems * 4
    if data_bytes > self.data_cap:
      self.d_data = cuda.mem_alloc(data_bytes)
      self.data_cap = data_bytes

    param_bytes = row_size * 4
    if param_bytes > self.param_cap:
      self.d_gamma = cuda.mem_alloc(param_bytes)
      self.d_beta = cuda.mem_alloc(param_bytes)
      self.param_cap = param_bytes


_ws = _LayerNormWorkspace()


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
  """
  Apply Layer Normalization to each row of the input matrix.

  Parameters
  ----------
  input : list or numpy.ndarray of float
      Flattened matrix in row-major order. Its length must be divisible by row_size.
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
      Flattened matrix of the same shape as input, containing the row-wise
      normalized results.
  """
  in_arr = np.ascontiguousarray(input, dtype=np.float32)
  total_elems = in_arr.size
  if total_elems == 0 or row_size <= 0:
    return in_arr.copy()

  gamma_arr = np.ascontiguousarray(gamma, dtype=np.float32)
  beta_arr = np.ascontiguousarray(beta, dtype=np.float32)
  row_count = total_elems // row_size

  _ws.init_kernels()
  _ws.ensure_buffers(total_elems, row_size)

  cuda.memcpy_htod_async(_ws.d_data, in_arr, _ws.stream)
  cuda.memcpy_htod_async(_ws.d_gamma, gamma_arr, _ws.stream)
  cuda.memcpy_htod_async(_ws.d_beta, beta_arr, _ws.stream)

  if row_size < 4:
    block_size = 256
    grid_size = (row_count + block_size - 1) // block_size
    _ws.kernel_small(
      _ws.d_data,
      _ws.d_gamma,
      _ws.d_beta,
      np.int32(row_count),
      np.int32(row_size),
      np.float32(eps),
      block=(block_size, 1, 1),
      grid=(grid_size, 1, 1),
      stream=_ws.stream,
    )
  else:
    vec_row_size = row_size // 4
    block_size = 256
    while block_size > vec_row_size and block_size > 32:
      block_size >>= 1

    _ws.kernel_vec4(
      _ws.d_data,
      _ws.d_gamma,
      _ws.d_beta,
      np.int32(vec_row_size),
      np.float32(eps),
      block=(block_size, 1, 1),
      grid=(row_count, 1, 1),
      stream=_ws.stream,
    )

  output = np.empty(in_arr.shape, dtype=np.float32)
  cuda.memcpy_dtoh_async(output, _ws.d_data, _ws.stream)
  _ws.stream.synchronize()

  return output