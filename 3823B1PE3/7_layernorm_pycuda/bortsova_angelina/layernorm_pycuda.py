import math
import numpy as np
from numba import cuda, float32

ROW_THREADS = 256

@cuda.jit
def collect_statistics(values, statistics, row_size, epsilon):
    row = cuda.blockIdx.x
    thread = cuda.threadIdx.x
    row_start = row * row_size
    anchor = values[row_start]
    means = cuda.shared.array(ROW_THREADS, float32)
    squared_deviations = cuda.shared.array(ROW_THREADS, float32)
    mean = float32(0.0)
    squared_sum = float32(0.0)
    sample_count = 0

    for column in range(thread, row_size, cuda.blockDim.x):
        value = values[row_start + column] - anchor
        sample_count += 1
        difference = value - mean
        mean += difference / float32(sample_count)
        squared_sum += difference * (value - mean)

    means[thread] = mean
    squared_deviations[thread] = squared_sum
    cuda.syncthreads()

    stride = cuda.blockDim.x // 2
    while stride > 0:
        if thread < stride:
            difference = means[thread + stride] - means[thread]
            means[thread] += difference * float32(0.5)
            squared_deviations[thread] += squared_deviations[thread + stride] + difference * difference * float32(sample_count) * float32(0.5)
        cuda.syncthreads()
        sample_count *= 2
        stride //= 2

    if thread == 0:
        variance = squared_deviations[0] / float32(row_size)
        statistics[row, 0] = anchor
        statistics[row, 1] = means[0]
        statistics[row, 2] = float32(1.0) / math.sqrt(variance + epsilon)

@cuda.jit
def normalize_elements(values, gamma, beta, statistics, row_size):
    index = cuda.grid(1)
    if index < values.size:
        row = index // row_size
        column = index % row_size
        centered = (values[index] - statistics[row, 0]) - statistics[row, 1]
        values[index] = centered * statistics[row, 2] * gamma[column] + beta[column]

def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    values = np.ascontiguousarray(input, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return np.empty(0, dtype=np.float32)

    row_count = values.size // row_size
    device_values = cuda.to_device(values)
    device_gamma = cuda.to_device(np.ascontiguousarray(gamma, dtype=np.float32))
    device_beta = cuda.to_device(np.ascontiguousarray(beta, dtype=np.float32))
    statistics = cuda.device_array((row_count, 3), dtype=np.float32)
    collect_statistics[row_count, min(ROW_THREADS, row_size)](device_values, statistics, row_size, np.float32(eps))
    block_count = (values.size + ROW_THREADS - 1) // ROW_THREADS
    normalize_elements[block_count, ROW_THREADS](device_values, device_gamma, device_beta, statistics, row_size)
    return device_values.copy_to_host()
