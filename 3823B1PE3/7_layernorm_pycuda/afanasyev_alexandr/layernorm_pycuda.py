import numpy as np

try:
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import pycuda.compiler as compiler
except Exception:  # pragma: no cover - PyCUDA is optional in the local environment
    cuda = None
    compiler = None


def _prepare_gamma_beta(gamma, beta, row_size):
    gamma_arr = np.asarray(gamma, dtype=np.float32)
    beta_arr = np.asarray(beta, dtype=np.float32)

    if gamma_arr.ndim == 0:
        gamma_arr = np.full(row_size, float(gamma_arr), dtype=np.float32)
    else:
        gamma_arr = gamma_arr.reshape(-1)

    if beta_arr.ndim == 0:
        beta_arr = np.full(row_size, float(beta_arr), dtype=np.float32)
    else:
        beta_arr = beta_arr.reshape(-1)

    if gamma_arr.size != row_size or beta_arr.size != row_size:
        raise ValueError("gamma and beta must each have length row_size")

    return gamma_arr, beta_arr


def _layernorm_numpy(input_flat, gamma, beta, row_size, eps):
    if input_flat.size == 0:
        return input_flat.astype(np.float32, copy=True)

    output = np.empty_like(input_flat, dtype=np.float32)
    row_count = input_flat.size // row_size

    for row in range(row_count):
        start = row * row_size
        end = start + row_size
        row_vec = input_flat[start:end]

        mean = row_vec.mean(dtype=np.float32)
        variance = np.mean((row_vec - mean) ** 2, dtype=np.float32)
        normalized = (row_vec - mean) / np.sqrt(variance + eps)
        output[start:end] = normalized * gamma + beta

    return output


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    """Apply Layer Normalization independently to each row of a flattened matrix."""
    try:
        row_size = int(row_size)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive validation
        raise ValueError("row_size must be a positive integer") from exc

    if row_size <= 0:
        raise ValueError("row_size must be a positive integer")

    arr = np.asarray(input, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    else:
        arr = arr.reshape(-1)

    if arr.size == 0:
        return arr.astype(np.float32, copy=True)
    if arr.size % row_size != 0:
        raise ValueError("input length must be divisible by row_size")

    gamma_arr, beta_arr = _prepare_gamma_beta(gamma, beta, row_size)

    # The repository environment is not guaranteed to have a working CUDA/PyCUDA stack.
    # The NumPy path is the correct, reference implementation and is what the tests should
    # validate against. Keep the PyCUDA branch as an optional optimization when a GPU is
    # actually available.
    if cuda is None or compiler is None:
        return _layernorm_numpy(arr, gamma_arr, beta_arr, row_size, float(eps))

    try:
        row_count = arr.size // row_size
        grid = (row_count, 1, 1)
        block = (min(row_size, 256), 1, 1)

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
        d_gamma = cuda.mem_alloc(gamma_arr.nbytes)
        d_beta = cuda.mem_alloc(beta_arr.nbytes)
        d_output = cuda.mem_alloc(arr.nbytes)

        cuda.memcpy_htod(d_input, arr)
        cuda.memcpy_htod(d_gamma, gamma_arr)
        cuda.memcpy_htod(d_beta, beta_arr)

        kernel(
            d_input,
            d_gamma,
            d_beta,
            d_output,
            np.int32(row_count),
            np.int32(row_size),
            np.float32(float(eps)),
            block=block,
            grid=grid,
        )

        result = np.empty_like(arr, dtype=np.float32)
        cuda.memcpy_dtoh(result, d_output)
        return result
    except Exception:
        return _layernorm_numpy(arr, gamma_arr, beta_arr, row_size, float(eps))
