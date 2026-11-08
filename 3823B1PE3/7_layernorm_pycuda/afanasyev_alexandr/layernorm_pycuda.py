import numpy as np

try:
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray
    import pycuda.compiler as compiler
    from pycuda.tools import DeviceData
except Exception:  # pragma: no cover
    cuda = None
    gpuarray = None
    compiler = None
    DeviceData = None


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
    try:
        row_size = int(row_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("row_size must be a positive integer") from exc

    if row_size <= 0:
        raise ValueError("row_size must be a positive integer")

    arr = np.asarray(input, dtype=np.float32)
    gamma = np.asarray(gamma, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)

    if arr.ndim != 1:
        arr = arr.ravel()
    if arr.size == 0:
        return arr.copy()
    if arr.size % row_size != 0:
        raise ValueError("input length must be divisible by row_size")
    if gamma.size != row_size or beta.size != row_size:
        raise ValueError("gamma and beta must each have length row_size")

    if cuda is None or compiler is None:
        row_count = arr.size // row_size
        out = np.empty_like(arr, dtype=np.float32)
        for r in range(row_count):
            start = r * row_size
            end = start + row_size
            x = arr[start:end]
            mean = x.mean(dtype=np.float32)
            var = ((x - mean) ** 2).mean(dtype=np.float32)
            normalized = (x - mean) / np.sqrt(var + eps)
            out[start:end] = normalized * gamma + beta
        return out

    row_count = arr.size // row_size
    block = 256
    grid = (row_count + block - 1) // block
    if row_count == 0:
        return arr.copy()

    kernel_source = r"""
    __global__ void layernorm_kernel(const float *input,
                                    const float *gamma,
                                    const float *beta,
                                    float *output,
                                    int row_count,
                                    int row_size,
                                    float eps) {
        int row = blockIdx.x;
        if (row >= row_count) {
            return;
        }

        int offset = row * row_size;
        float mean = 0.0f;
        for (int i = 0; i < row_size; ++i) {
            mean += input[offset + i];
        }
        mean /= float(row_size);

        float variance = 0.0f;
        for (int i = 0; i < row_size; ++i) {
            float diff = input[offset + i] - mean;
            variance += diff * diff;
        }
        variance /= float(row_size);

        float inv_std = rsqrtf(variance + eps);
        for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
            int idx = offset + i;
            float normalized = (input[idx] - mean) * inv_std;
            output[idx] = normalized * gamma[i] + beta[i];
        }
    }
    """

    module = compiler.SourceModule(kernel_source)
    kernel = module.get_function("layernorm_kernel")

    d_input = cuda.mem_alloc(arr.nbytes)
    d_gamma = cuda.mem_alloc(gamma.nbytes)
    d_beta = cuda.mem_alloc(beta.nbytes)
    d_output = cuda.mem_alloc(arr.nbytes)

    cuda.memcpy_htod(d_input, arr)
    cuda.memcpy_htod(d_gamma, gamma)
    cuda.memcpy_htod(d_beta, beta)

    kernel(d_input, d_gamma, d_beta, d_output, np.int32(row_count), np.int32(row_size), np.float32(eps),
           block=(block, 1, 1), grid=(grid, 1, 1))

    out = np.empty_like(arr, dtype=np.float32)
    cuda.memcpy_dtoh(out, d_output)

    return out
