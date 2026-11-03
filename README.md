# Content
- [How To](#how-to)
- [Configuration](#configuration)
- [Time Measurement](#time-measurement)
- [Tasks](#tasks)
- [Results](#results)

# How To
1. Create [github](https://github.com/) account (if not exists);
2. Make sure SSH clone & commit is working ([Connecting to GitHub with SSH](https://docs.github.com/en/authentication/connecting-to-github-with-ssh));
3. Fork this repo (just click **Fork** button on the top of the page, detailed instructions [here](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project))
4. Clone your forked repo into your local machine, use your user instead of `username`:
```sh
git clone git@github.com:username/gpu-2026.git
cd gpu-2026
```
5. Go to your group folder, e.g.:
```sh
cd 3823B1FI1
```
6. Go to needed task folder, e.g.:
```sh
cd 1_gelu_omp
```
7. Create new folder with your surname and name (**make sure it's the same for all tasks**), e.g.:
```sh
mkdir petrov_ivan
```
8. Copy your task source/header files (including main program) into this folder (use `copy` instead of `cp` on Windows), e.g.:
```sh
cd petrov_ivan
cp /home/usr/lab/*.cpp .
cp /home/usr/lab/*.h .
```
8. Push your sources to github repo, e.g.:
```sh
cd ..
git add .
git commit -m "1_gelu_omp task"
git push
```
9. Go to your repo in browser, click **Contribute** button on the top of page, then **Open pull request**. Provide meaningfull request title and description, then **Create pull request** (see details [here](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project)).
10. Go to Pull Requests [page](https://github.com/avgorshk/gpu-2025/pulls) in course repo, find your pull request and check if there are no any merge conflicts occur. If merge conflicts happen - resolve it following the instruction provided by github.

# Time Measurement
The following scheme is used to measure task execution time:
```cpp
int main() {
    // ...

    // Warming-up
    Task(input, size);

    // Performance Measuring
    std::vector<double> time_list;
    for (int i = 0; i < 4; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        Task(input, size);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        time_list.push_back(duration.count());
    }
    double time = *std::min_element(time_list.begin(), time_list.end());

    // ...
}
```

# Configuration
- CPU: Intel Core i5 12600K (4 cores, 4 threads)
- RAM: 16 GB
- GPU: NVIDIA RTX 4060 (8 GB)
- OS:  Ubuntu 22.04.3 LTS
- Host Compiler: GCC 11.4.0 (C++17)
- CUDA: 13.3

# Tasks
## Task #1: OpenMP GELU Implementation
The **Gaussian Error Linear Unit (GELU)** is an activation function frequently used in Deep Neural Networks (DNNs) and can be thought of as a smoother ReLU.

To approximate GELU function, use the following formula:

GELU(x) =  $0.5x(1 + tanh(\sqrt{2 / \pi}(x + 0.044715 * x^3)))$

Implement the function with the following interface in C++:
```cpp
std::vector<float> GeluOMP(const std::vector<float>& input);
```
Size of result vector should be the same as for `input`. Use OpenMP technology to make your function parallel & fast.

Two files are expected to be uploaded:
- gelu_omp.h
```cpp
#ifndef __GELU_OMP_H
#define __GELU_OMP_H

#include <vector>

std::vector<float> GeluOMP(const std::vector<float>& input);

#endif // __GELU_OMP_H
```
- gelu_omp.cpp
```cpp
#include "gelu_omp.h"

std::vector<float> GeluOMP(const std::vector<float>& input) {
    // Place your implementation here
}
```
**Performance Hints:**
 - better formula to compute GELU, e.g. replace *tanh()* with *exp()*;
 - loop unrolling;
 - loop vectorization;
 - vector allocation and computations in different threads *(Windows only)*.

## Task #2: CUDA GELU Implementation
Implement the function with the following interface in CUDA C++ using the formula described above:
```cpp
std::vector<float> GeluCUDA(const std::vector<float>& input);
```
Size of result vector should be the same as for `input`. Use CUDA technology to make your function work on NVIDIA GPU. Try to make it fast.

Two files are expected to be uploaded:
- gelu_cuda.h
```cpp
#ifndef __GELU_CUDA_H
#define __GELU_CUDA_H

#include <vector>

std::vector<float> GeluCUDA(const std::vector<float>& input);

#endif // __GELU_CUDA_H
```
- gelu_cuda.cu
```cpp
#include "gelu_cuda.h"

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    // Place your implementation here
}
```
**Performance Hints:**
 - overlap host memory allocation and CUDA computations;
 - allocate and free device memory once;
 - use better formula to compute GELU, e.g. replace *tanh()* with *exp()*.

## Task #3: Naive Matrix Multiplication using CUDA
General matrix multiplication (GEMM) is a very basic and broadly used linear algebra operation applied in high performance computing (HPC), statistics, deep learning and other domains. There are a lot of GEMM algorithms with different mathematical complexity form $O(n^3)$ for naive and block approaches to $O(n^{2.371552})$ for the method descibed by Williams et al. in 2024 [[1](https://epubs.siam.org/doi/10.1137/1.9781611977912.134)]. But despite a variety of algorithms with low complexity, block matrix multiplication remains the most used implementation in practice since it fits to modern HW better.

To start learning matrix multiplication smoother, let us start with naive approach here. To compute matrix multiplication result C for matricies A and B, where C = A * B and the size for all matricies are $n*n$, one should use the following formula for each element of C (will consider only square matricies for simplicity):

$c_{ij}=\sum_{k=1}^na_{ik}b_{kj}$

In this task one should implement naive approach for matrix multiplication in CUDA trying to make it fast enough *(pay attention to global memory accesses in your code)*.

Each matrix must be stored in a linear array by rows, so that `a.size()==n*n`. Function takes two matricies and their size as inputs, and returns result matrix also stored by rows.

For simplicity, let's consider matrix size is always power of 2.

Two files are expected to be uploaded:
- naive_gemm_cuda.h:
```cpp
#ifndef __NAIVE_GEMM_CUDA_H
#define __NAIVE_GEMM_CUDA_H

#include <vector>

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n);

#endif // __NAIVE_GEMM_CUDA_H
```
- naive_gemm_cuda.cu:
```cpp
#include "naive_gemm_cuda.h"

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    // Place your implementation here
}
```
**Performance Hints:**
 - warp-friendly memory accesses;
 - multiple elements per warp processing;
 - loop unrolling and memory load vectorization;
 - block size selection;
 - overlap host memory allocation and CUDA computations.

## Task #4: Block Matrix Multiplication using CUDA
In real applications block-based approach for matrix multiplication can get multiple times faster execution comparing with naive version due to cache friendly approach. To prove this in practice, implement such a version in C++ using OpenMP.

In block version algorithm could be divided into three stages:
1. Split matricies into blocks (block size normally affects performance significantly so choose it consciously);
2. Multiply two blocks to get partial result;
3. Replay step 2 for all row/column blocks accumulating values into a single result block.

From math perspective, block matrix multiplication could be described by the following formula, where $C_{IJ}$, $A_{IK}$ and $B_{KJ}$ are sub-matricies with the size $block\_size*block\_size$:

$C_{IJ}=\sum_{k=1}^{block_count}A_{IK}B_{KJ}$

Each matrix must be stored in a linear array by rows, so that `a.size()==n*n`. Function takes two matricies and their size as inputs, and returns result matrix also stored by rows.

In CUDA C++ block-based approach looks similar. But to get better performance one should use CUDA shared memory to store each particular block while computations. With this consideration, algorithm will be the following:
1. A single CUDA block should compute a single block of result matrix C, a single CUDA thread - a single matrix C element;
2. For each A block in a row and B block in a column:
    1. Load A block into shared memory;
    2. Load B block into shared memory;
    3. Synchronize over all threads in block;
    4. Compute BlockA * BlockB and accumulate into C block in shared memory;
    5. Synchronize over all threads in block;
3. Dump block C from shared to global memory.

For simplicity, let's consider matrix size is always power of 2.

Two files are expected to be uploaded:
- block_gemm_cuda.h:
```cpp
#ifndef __BLOCK_GEMM_CUDA_H
#define __BLOCK_GEMM_CUDA_H

#include <vector>

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n);

#endif // __BLOCK_GEMM_CUDA_H
```
- block_gemm_cuda.cu:
```cpp
#include "block_gemm_cuda.h"

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    // Place your implementation here
}
```
**Performance Hints:**
 - shared memory usage to store matrix block;
 - warp-friendly memory accesses;
 - multiple elements per warp processing;
 - loop unrolling and memory load vectorization;
 - block size selection;
 - overlap host memory allocation and CUDA computations.

## Task #5: Matrix Multiplication using cuBLAS
The most performant way to multiply two matrices on particular hardware is to use vendor-provided library for this purpose. In CUDA it's [cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html). Try to use cuBLAS API to implement general matrix multiplication in most performant way.

Each matrix must be stored in a linear array by rows, so that `a.size()==n*n`. Function takes two matricies and their size as inputs, and returns result matrix also stored by rows.

For simplicity, let's consider matrix size is always power of 2.

Note, that in cuBLAS API matrix is expected to be stored by columns, so additional transpose may be required.

Two files are expected to be uploaded:
- gemm_cublas.h:
```cpp
#ifndef __GEMM_CUBLAS_H
#define __GEMM_CUBLAS_H

#include <vector>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n);

#endif // __GEMM_CUBLAS_H
```
- gemm_cublas.cu:
```cpp
#include "gemm_cublas.h"

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    // Place your implementation here
}
```
**Performance Hints:**
 - overlap host memory allocation and CUDA computations;
 - avoid redundant device memory allocation.

## Task #6: CUDA Softmax Implementation
The **softmax** function is a fundamental operation in machine learning, often used to convert a vector of raw scores into a probability distribution. For an input vector $x$ of length $N$, the softmax is defined element-wise as:

Softmax(x) = $e^{x_i}/(\sum_{j=1}^ne^{x_j})$ for $i=1,..,N$

When the input is a matrix, softmax is applied independently to each row.

To make the computation numerically stable in floating-point arithmetic, the following equivalent formula is used in practice:

Softmax(x) = $e^{(x_i-row\_max)}/(\sum_{j=1}^ne^{(x_j-row\_max)})$ for $i=1,..,N$

Here $row\_max$ is $max(x_i)$ for $i=1,..,N$, normally computed independently for each row in matrix.

Implement the function with the following interface in C++ using CUDA:
```cpp
std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count);
```
Note the following:
- the parameter input holds the matrix elements in row‑major order (all elements of row 0, then row 1, etc.);
- the number of rows is given by `row_count`;
- the number of columns (size of each row) can be derived as `row_size = input.size() / row_count` (it is guaranteed that `input.size()` is divisible by row_count);
- the function must compute softmax for each row independently and return a vector of the same size containing the row‑wise softmax results.

Use CUDA to parallelize the computation. The implementation should be efficient – consider using shared memory for per‑row reductions and exponentiations.

For simplicity, let's consider matrix sizes are always power of 2.

Two files are expected to be uploaded:
- softmax_cuda.h:
```cpp
#ifndef SOFTMAX_CUDA_H
#define SOFTMAX_CUDA_H

#include <vector>

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count);

#endif // SOFTMAX_CUDA_H
```
- softmax_cuda.cu:
```cpp
#include "softmax_cuda.h"

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    // Place your implementation here
}
```
**Performance Hints:**
 - overlap host memory allocation and CUDA computations;
 - use registers and/or shared memory to cache input values.

## Task #7: Layer Norm Implementation in PyCUDA
Layer Normalization (**LayerNorm**) is a widely used technique in deep learning that normalizes activations across the feature dimension for each sample independently. For an input vector x of length N (the features of one sample), LayerNorm is defined as:

$$x'_i=(x_i-\mu)/\sqrt{\sigma^2+\epsilon}$$
$$y_i=\gamma_ix'_i+\beta_i$$

where:
- $\mu=1/N*\sum_{j=1}^Nx_j$ is the mean of the features;
- $\sigma^2=1/N*\sum_{j=1}^N(x_j-\mu)^2$ is the variance;
- $\epsilon$ is a small constant for numerical stability (e.g. $10^-5$);
- $\gamma$ and $\beta$ are learnable parameters (vectors of length N) that scale and shift the normalized output.

When the input is a matrix (batch of samples), LayerNorm is applied independently to each row.

To complete the task, one have to implement the following function in PyCUDA, the only file is expected to be upload:
- layernorm_pycuda.py
```py
import numpy as np

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
    # TODO: Implement using PyCUDA
    pass
```

For simplicity, let's consider `row_size` is power of 2. Target data type is float32.
One may use numba or C strings to write CUDA kernels.

## Task #8: OpenCL GELU Implementation
Implement GELU function with the following interface in OpenCL using the formula described in task #1:
```cpp
std::vector<float> GeluOCL(const std::vector<float>& input, int platform);
```
Size of result vector should be the same as for `input`. Use OpenCL technology to make your function work on NVIDIA GPU. Try to make it fast.

Use `CL_DEVICE_GPU` flag to choose GPU device. Use `platform` platform and `0` device. Store your OpenCL kernel in a string constant.

Two files are expected to be uploaded:
- gelu_ocl.h
```cpp
#ifndef __GELU_OCL_H
#define __GELU_OCL_H

#include <vector>

std::vector<float> GeluOCL(const std::vector<float>& input, int platform);

#endif // __GELU_OCL_H
```
- gelu_ocl.cpp
```cpp
#include "gelu_ocl.h"

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    // Place your implementation here
}
```
**Performance Hints:**
 - perform OpenCL boilerplate code once;
 - use better formula to compute GELU, e.g. replace *tanh()* with *exp()*;
 - overlap host memory allocation and GPU computations.

# Results
## 1_gelu_omp (134217728 elements)
|Group|Name|Result|Rank|
|-----|----|------|----|
|3823B1PE1|makoveeva_sofia|0.0841|14|
|3823B1PE1|shilin_nikita|0.0864|6|
|3823B1FI1|dorofeev_ivan|0.1488|4|
|**FAST**|**FAST**|**0.1741**|**-**|
|3823B1PE3|rastvorov_kirill|0.2228|7|
|3823B1PE2|zyuzin_nikita|0.2249|9|
|3823B1PE2|akhmetov_daniil|0.2257|10|
|3823B1PE1|rusakova_aleksandra|0.2258|3|
|3823B1PE1|melnik_ivan|0.2259|11|
|3823B1PE1|tabalaev_anton|0.2269|13|
|3823B1PE4|remizov_kirill|0.2275|3|
|3823B1FI2|volkov_aleksei|0.2276|4|
|3823B1PE1|perepelkin_iaroslav|0.2285|8|
|3823B1PE4|dilshodov_adkham|0.2292|2|
|3823B1FI2|ashihmin_daniil|0.2293|5|
|3823B1FI2|rozenberg_alexey|0.2303|3|
|3823B1PE4|karpich_ivan|0.2313|11|
|3823B1PE4|terekhov_dmitriy|0.2329|8|
|3823B1FI3|lifanov_kirill|0.2333|1|
|3823B1PE2|olesnitskiy_vladimir|0.2337|12|
|3823B1PE1|ivanova_polina|0.2358|12|
|3823B1PE3|bortsova_angelina|0.2359|3|
|3823B1PE2|viderman_aleksandra|0.2361|7|
|3823B1PE1|otcheskov_semyon|0.2367|5|
|3823B1PE3|balchunayte_zlata|0.2377|8|
|3823B1PE1|tsibareva_ekaterina|0.2378|4|
|3823B1PE3|dergachev_arseniy|0.2386|1|
|3823B1PE2|kolotukhin_alexander|0.2394|3|
|3823B1PE1|lazareva_ann|0.2397|15|
|3823B1FI1|zenin_anton|0.2408|1|
|3823B1FI2|sannikov_ivan|0.2410|2|
|3823B1PE1|kondrashova_victoria|0.2416|10|
|3823B1PE2|ovchinnikov_matvey|0.2446|11|
|3823B1PE3|marin_lev|0.2457|4|
|3823B1PE4|peryashkin_vasily|0.2506|4|
|3823B1FI1|krykov_egor|0.2522|3|
|3823B1PE2|vasiliev_mikhail|0.2529|5|
|3823B1PE4|trofimov_nikita|0.2533|10|
|3823B1PE3|batkov_filipp|0.2535|2|
|3823B1PE2|sinev_artem|0.2567|1|
|3823B1FI1|kosolapov_vitaliy|0.2568|2|
|3823B1PE2|tsyplakov_kirill|0.2570|8|
|3823B1PE1|zhurin_ivan|0.2580|7|
|3823B1PE2|kotelnikova_anastasia|0.2591|6|
|3823B1PE2|klimenko_vladislav|0.2611|2|
|3823B1PE1|golovanov_daniil|0.2612|9|
|3823B1PE1|redkina_alina|0.2634|2|
|3823B1FI2|chyokotov_alexey|0.2675|1|
|3823B1PE3|borunov_vladislav|0.2710|5|
|3823B1PE1|zyazeva_svetlana|0.2734|17|
|3823B1PE2|zorin_danila_artemovich|0.4702|4|
|3823B1PE1|morozov_nikita|0.4711|1|
|3823B1PE3|afanasyev_alexandr|0.4714|9|
|3823B1PE4|baranov_andrey|0.4798|6|
|3823B1PE1|chernov_timur|0.4801|16|
|3823B1PE4|tsarkov_klim|0.4829|5|
|3823B1PE3|ermakov_alexei|0.4845|6|
|3823B1PE1|popova_elizaveta|0.4898|18|
|3823B1PE4|zaharov_gleb|0.4929|1|
|3823B1PE4|romanov_michail|0.4937|7|
|3823B1PE4|safronov_maksim|0.4943|9|
|**REF**|**REF**|**0.5440**|**-**|
|3823B1PE4|urin_oleg|TEST FAILED|-|

## 2_gelu_cuda (134217728 elements)
|Group|Name|Result|Rank|
|-----|----|------|----|
|3823B1FI1|dorofeev_ivan|0.1596|3|
|3823B1PE3|ermakov_alexei|0.1793|5|
|3823B1PE4|karpich_ivan|0.1973|11|
|3823B1PE2|ovchinnikov_matvey|0.2001|12|
|3823B1PE2|vasiliev_mikhail|0.2001|5|
|3823B1PE4|remizov_kirill|0.2018|3|
|3823B1PE3|borunov_vladislav|0.2026|6|
|3823B1PE4|dilshodov_adkham|0.2056|2|
|3823B1FI2|volkov_aleksei|0.2063|4|
|3823B1PE2|zyuzin_nikita|0.2065|9|
|**FAST**|**FAST**|**0.2067**|**-**|
|3823B1PE4|peryashkin_vasily|0.2078|4|
|3823B1PE2|viderman_aleksandra|0.2078|6|
|3823B1FI1|zenin_anton|0.2104|1|
|3823B1PE4|romanov_michail|0.2108|7|
|3823B1PE1|shilin_nikita|0.2111|5|
|3823B1PE3|dergachev_arseniy|0.2192|1|
|3823B1PE3|rastvorov_kirill|0.2380|7|
|3823B1PE2|akhmetov_daniil|0.2381|10|
|3823B1PE1|redkina_alina|0.2390|1|
|3823B1PE2|kotelnikova_anastasia|0.2399|4|
|3823B1PE2|sinev_artem|0.2406|1|
|3823B1PE1|perepelkin_iaroslav|0.2427|7|
|3823B1PE2|kolotukhin_alexander|0.2435|7|
|3823B1PE1|rusakova_aleksandra|0.2437|3|
|3823B1PE1|melnik_ivan|0.2444|10|
|3823B1FI1|krykov_egor|0.2449|2|
|3823B1PE3|marin_lev|0.2451|4|
|3823B1PE1|golovanov_daniil|0.2462|8|
|3823B1PE1|popova_elizaveta|0.2490|15|
|3823B1PE3|bortsova_angelina|0.2491|3|
|3823B1PE1|kondrashova_victoria|0.2491|9|
|3823B1PE1|otcheskov_semyon|0.2500|4|
|3823B1PE2|olesnitskiy_vladimir|0.2508|11|
|3823B1PE1|lazareva_ann|0.2511|13|
|3823B1PE1|chernov_timur|0.2524|14|
|3823B1PE1|tabalaev_anton|0.2534|12|
|3823B1FI2|rozenberg_alexey|0.2537|1|
|3823B1PE4|safronov_maksim|0.2545|9|
|3823B1FI2|sannikov_ivan|0.2562|3|
|3823B1PE4|zaharov_gleb|0.2575|1|
|3823B1PE4|terekhov_dmitriy|0.2626|8|
|3823B1PE4|baranov_andrey|0.2641|6|
|3823B1PE2|zorin_danila_artemovich|0.2690|2|
|3823B1FI3|lifanov_kirill|0.2693|1|
|3823B1FI2|chyokotov_alexey|0.2710|2|
|**REF**|**REF**|**0.2717**|**-**|
|3823B1PE2|tsyplakov_kirill|0.2721|8|
|3823B1PE3|afanasyev_alexandr|0.2722|9|
|3823B1PE2|klimenko_vladislav|0.2741|3|
|3823B1PE4|tsarkov_klim|0.2744|5|
|3823B1PE3|batkov_filipp|0.2750|2|
|3823B1PE1|tsibareva_ekaterina|0.2788|2|
|3823B1PE1|ivanova_polina|0.2793|11|
|3823B1PE3|balchunayte_zlata|0.2797|8|
|3823B1PE4|trofimov_nikita|0.2801|10|
|3823B1PE1|zhurin_ivan|0.3955|6|
|3823B1PE4|urin_oleg|TEST FAILED|-|

## 3_naive_gemm_cuda (4096 elements)
|Group|Name|Result|Rank|
|-----|----|------|----|
|3823B1FI1|dorofeev_ivan|0.0268|2|
|3823B1PE3|ermakov_alexei|0.0364|5|
|3823B1PE4|dilshodov_adkham|0.0380|1|
|3823B1PE2|viderman_aleksandra|0.0393|5|
|3823B1PE1|melnik_ivan|0.0407|14|
|3823B1PE4|karpich_ivan|0.0415|12|
|3823B1PE3|rastvorov_kirill|0.0420|6|
|3823B1FI2|volkov_aleksei|0.0440|4|
|3823B1PE4|remizov_kirill|0.0444|4|
|3823B1PE3|dergachev_arseniy|0.0479|1|
|3823B1PE3|borunov_vladislav|0.0483|7|
|3823B1PE1|shilin_nikita|0.0487|4|
|3823B1FI2|sannikov_ivan|0.0489|3|
|3823B1FI2|rozenberg_alexey|0.0507|1|
|3823B1PE2|kolotukhin_alexander|0.0509|8|
|3823B1PE3|batkov_filipp|0.0619|4|
|3823B1PE3|marin_lev|0.0636|2|
|3823B1PE2|sinev_artem|0.0653|6|
|3823B1PE1|otcheskov_semyon|0.0664|3|
|3823B1PE1|tabalaev_anton|0.0685|11|
|3823B1PE2|akhmetov_daniil|0.0688|10|
|3823B1PE1|perepelkin_iaroslav|0.0692|5|
|3823B1PE1|redkina_alina|0.0712|1|
|**FAST**|**FAST**|**0.0732**|**-**|
|3823B1PE1|ivanova_polina|0.0742|8|
|3823B1PE2|vasiliev_mikhail|0.0750|4|
|3823B1PE2|zyuzin_nikita|0.0753|9|
|3823B1PE1|kondrashova_victoria|0.0783|9|
|3823B1PE2|tsyplakov_kirill|0.0791|7|
|3823B1PE1|popova_elizaveta|0.0802|15|
|3823B1PE1|rusakova_aleksandra|0.0927|2|
|3823B1PE3|bortsova_angelina|0.0972|3|
|3823B1PE1|lazareva_ann|0.1024|12|
|3823B1PE4|trofimov_nikita|0.1098|11|
|3823B1PE2|olesnitskiy_vladimir|0.1111|11|
|3823B1PE2|klimenko_vladislav|0.1138|1|
|3823B1FI1|zenin_anton|0.1349|1|
|3823B1PE4|peryashkin_vasily|0.1430|5|
|3823B1FI3|lifanov_kirill|0.1437|1|
|3823B1PE2|kotelnikova_anastasia|0.1479|3|
|3823B1PE3|afanasyev_alexandr|0.1774|9|
|3823B1PE4|baranov_andrey|0.1791|6|
|3823B1PE4|tsarkov_klim|0.1795|7|
|3823B1PE1|golovanov_daniil|0.1801|10|
|3823B1PE4|safronov_maksim|0.1821|10|
|3823B1PE4|terekhov_dmitriy|0.1875|9|
|3823B1PE4|romanov_michail|0.1879|8|
|3823B1PE4|urin_oleg|0.1886|2|
|3823B1PE4|zaharov_gleb|0.1887|3|
|3823B1PE1|zhurin_ivan|0.1910|6|
|3823B1FI2|chyokotov_alexey|0.1913|2|
|3823B1PE3|balchunayte_zlata|0.1940|8|
|3823B1PE1|chernov_timur|0.1967|13|
|3823B1PE1|tsibareva_ekaterina|0.2035|7|
|3823B1PE2|zorin_danila_artemovich|0.2092|2|
|**REF**|**REF**|**0.5864**|**-**|

## 4_block_gemm_cuda (4096 elements)
|Group|Name|Result|Rank|
|-----|----|------|----|
|3823B1FI1|dorofeev_ivan|0.0259|2|
|3823B1PE3|rastvorov_kirill|0.0365|6|
|3823B1PE3|ermakov_alexei|0.0373|5|
|3823B1PE1|lazareva_ann|0.0377|12|
|3823B1PE1|shilin_nikita|0.0383|3|
|3823B1PE2|viderman_aleksandra|0.0386|5|
|3823B1PE4|karpich_ivan|0.0395|11|
|3823B1PE2|kolotukhin_alexander|0.0399|11|
|3823B1PE4|remizov_kirill|0.0415|2|
|3823B1PE4|dilshodov_adkham|0.0417|1|
|3823B1PE3|dergachev_arseniy|0.0443|1|
|3823B1PE3|marin_lev|0.0457|2|
|3823B1PE1|melnik_ivan|0.0489|14|
|3823B1PE3|bortsova_angelina|0.0591|3|
|3823B1PE2|akhmetov_daniil|0.0591|9|
|3823B1FI1|zenin_anton|0.0593|1|
|3823B1PE1|perepelkin_iaroslav|0.0599|5|
|3823B1PE1|tabalaev_anton|0.0599|11|
|3823B1PE2|vasiliev_mikhail|0.0608|4|
|3823B1PE1|otcheskov_semyon|0.0611|4|
|3823B1PE1|redkina_alina|0.0631|1|
|3823B1PE4|zaharov_gleb|0.0658|4|
|3823B1PE1|ivanova_polina|0.0674|9|
|3823B1PE2|olesnitskiy_vladimir|0.0698|10|
|**FAST**|**FAST**|**0.0701**|**-**|
|3823B1PE4|peryashkin_vasily|0.0724|5|
|3823B1PE1|tsibareva_ekaterina|0.0775|8|
|3823B1PE1|popova_elizaveta|0.0792|15|
|3823B1PE2|sinev_artem|0.0815|6|
|3823B1PE2|klimenko_vladislav|0.0842|1|
|3823B1PE4|romanov_michail|0.1328|7|
|3823B1PE4|urin_oleg|0.1329|3|
|3823B1PE4|safronov_maksim|0.1360|9|
|3823B1PE1|zhurin_ivan|0.1384|7|
|3823B1PE3|afanasyev_alexandr|0.1389|7|
|3823B1PE2|tsyplakov_kirill|0.1391|7|
|3823B1PE4|terekhov_dmitriy|0.1423|8|
|3823B1PE2|zyuzin_nikita|0.1439|8|
|3823B1PE3|batkov_filipp|0.1440|4|
|3823B1PE4|tsarkov_klim|0.1449|6|
|3823B1PE2|zorin_danila_artemovich|0.1461|2|
|3823B1PE2|kotelnikova_anastasia|0.1464|3|
|3823B1FI3|lifanov_kirill|0.1515|1|
|3823B1FI2|chyokotov_alexey|0.1528|1|
|3823B1PE1|golovanov_daniil|0.1536|6|
|3823B1PE1|kondrashova_victoria|0.1542|10|
|3823B1PE4|trofimov_nikita|0.1559|10|
|3823B1PE1|rusakova_aleksandra|0.1805|2|
|3823B1PE1|chernov_timur|0.2108|13|
|**REF**|**REF**|**0.3076**|**-**|

## 5_gemm_cublas (4096 elements)
|Group|Name|Result|Rank|
|-----|----|------|----|
|3823B1FI1|dorofeev_ivan|0.0261|2|
|3823B1PE3|ermakov_alexei|0.0269|4|
|3823B1PE4|karpich_ivan|0.0339|9|
|3823B1FI1|zenin_anton|0.0340|1|
|3823B1PE2|viderman_aleksandra|0.0344|6|
|3823B1PE3|marin_lev|0.0348|2|
|3823B1PE1|perepelkin_iaroslav|0.0349|3|
|3823B1PE4|romanov_michail|0.0354|5|
|3823B1PE1|tabalaev_anton|0.0354|12|
|3823B1PE3|bortsova_angelina|0.0356|3|
|3823B1PE4|dilshodov_adkham|0.0358|1|
|3823B1PE2|vasiliev_mikhail|0.0359|4|
|3823B1PE1|redkina_alina|0.0359|1|
|3823B1PE3|rastvorov_kirill|0.0360|5|
|3823B1PE4|remizov_kirill|0.0360|6|
|3823B1PE3|dergachev_arseniy|0.0363|1|
|3823B1PE1|otcheskov_semyon|0.0363|4|
|3823B1PE2|akhmetov_daniil|0.0364|8|
|3823B1PE1|rusakova_aleksandra|0.0370|2|
|3823B1PE2|olesnitskiy_vladimir|0.0374|9|
|3823B1PE2|zyuzin_nikita|0.0386|7|
|3823B1PE1|ivanova_polina|0.0399|7|
|3823B1PE4|zaharov_gleb|0.0412|2|
|3823B1PE4|terekhov_dmitriy|0.0412|8|
|3823B1PE4|peryashkin_vasily|0.0413|3|
|3823B1PE4|urin_oleg|0.0413|7|
|**FAST**|**FAST**|**0.0417**|**-**|
|3823B1PE1|zhurin_ivan|0.0430|13|
|3823B1PE2|zorin_danila_artemovich|0.0498|2|
|3823B1PE1|shilin_nikita|0.0503|6|
|3823B1PE2|kotelnikova_anastasia|0.0505|3|
|3823B1PE1|popova_elizaveta|0.0506|14|
|3823B1PE1|golovanov_daniil|0.0506|5|
|3823B1FI3|lifanov_kirill|0.0517|1|
|3823B1PE1|chernov_timur|0.0538|11|
|3823B1PE1|tsibareva_ekaterina|0.0539|9|
|3823B1PE2|sinev_artem|0.0557|5|
|3823B1PE2|klimenko_vladislav|0.0563|1|
|3823B1PE1|kondrashova_victoria|0.0567|8|
|3823B1PE4|tsarkov_klim|0.0583|4|
|**REF**|**REF**|**0.0591**|**-**|
|3823B1PE1|lazareva_ann|0.0600|10|
|3823B1PE3|afanasyev_alexandr|0.7127|6|

## 6_softmax_cuda (8192x16384 elements)
|Group|Name|Result|Rank|
|-----|----|------|----|
|3823B1FI1|dorofeev_ivan|0.1687|1|
|3823B1PE3|ermakov_alexei|0.1948|4|
|3823B1PE2|akhmetov_daniil|0.1990|8|
|3823B1PE3|rastvorov_kirill|0.2011|5|
|3823B1FI1|zenin_anton|0.2073|2|
|3823B1PE2|zyuzin_nikita|0.2090|7|
|3823B1PE4|karpich_ivan|0.2092|9|
|3823B1PE2|viderman_aleksandra|0.2107|6|
|3823B1PE4|peryashkin_vasily|0.2125|3|
|**FAST**|**FAST**|**0.2142**|**-**|
|3823B1PE1|otcheskov_semyon|0.2187|4|
|3823B1PE1|tabalaev_anton|0.2350|11|
|3823B1PE1|perepelkin_iaroslav|0.2351|3|
|3823B1PE2|sinev_artem|0.2370|5|
|3823B1PE3|bortsova_angelina|0.2396|2|
|3823B1PE2|vasiliev_mikhail|0.2422|4|
|3823B1PE4|romanov_michail|0.2432|5|
|3823B1PE1|redkina_alina|0.2432|1|
|3823B1PE3|marin_lev|0.2435|3|
|3823B1PE1|rusakova_aleksandra|0.2446|2|
|3823B1PE3|dergachev_arseniy|0.2466|1|
|3823B1PE1|shilin_nikita|0.2473|6|
|3823B1PE1|ivanova_polina|0.2474|7|
|3823B1PE4|dilshodov_adkham|0.2476|1|
|3823B1PE1|chernov_timur|0.2486|10|
|3823B1PE1|lazareva_ann|0.2489|9|
|3823B1PE4|remizov_kirill|0.2508|6|
|3823B1FI3|lifanov_kirill|0.2513|1|
|3823B1PE4|terekhov_dmitriy|0.2563|7|
|3823B1PE4|zaharov_gleb|0.2655|2|
|3823B1PE2|klimenko_vladislav|0.2685|1|
|**REF**|**REF**|**0.2707**|**-**|
|3823B1PE4|urin_oleg|0.2716|8|
|3823B1PE2|kotelnikova_anastasia|0.2728|3|
|3823B1PE1|kondrashova_victoria|0.2731|8|
|3823B1PE2|zorin_danila_artemovich|0.2745|2|
|3823B1PE4|tsarkov_klim|0.2753|4|
|3823B1PE1|popova_elizaveta|0.2759|13|
|3823B1PE1|tsibareva_ekaterina|0.2809|12|
|3823B1PE1|golovanov_daniil|0.2818|5|
|3823B1PE2|olesnitskiy_vladimir|0.2886|9|
|3823B1PE3|afanasyev_alexandr|0.2893|6|

## 7_layernorm_pycuda (8192x16384 elements)
|Group|Name|Result|Rank|
|-----|----|------|----|
|3823B1PE1|otcheskov_semyon|0.0460|3|
|3823B1PE2|vasiliev_mikhail|0.1370|5|
|3823B1PE4|peryashkin_vasily|0.1370|3|
|3823B1FI3|lifanov_kirill|0.1390|1|
|3823B1PE1|perepelkin_iaroslav|0.1420|4|
|3823B1PE1|ivanova_polina|0.1420|7|
|3823B1PE4|dilshodov_adkham|0.1440|1|
|3823B1FI1|dorofeev_ivan|0.1450|1|
|3823B1PE1|rusakova_aleksandra|0.1460|2|
|3823B1PE2|akhmetov_daniil|0.1470|8|
|3823B1PE2|viderman_aleksandra|0.1480|6|
|3823B1PE4|karpich_ivan|0.1490|5|
|3823B1PE1|chernov_timur|0.1500|10|
|3823B1PE1|redkina_alina|0.1510|1|
|3823B1FI1|zenin_anton|0.1510|2|
|3823B1PE3|rastvorov_kirill|0.1510|5|
|3823B1PE1|shilin_nikita|0.1520|5|
|3823B1PE2|kotelnikova_anastasia|0.1520|3|
|3823B1PE3|dergachev_arseniy|0.1540|1|
|3823B1PE3|bortsova_angelina|0.1570|2|
|3823B1PE3|marin_lev|0.1580|3|
|3823B1PE1|golovanov_daniil|0.1640|8|
|3823B1PE3|ermakov_alexei|0.1690|4|
|3823B1PE2|klimenko_vladislav|0.1730|1|
|3823B1PE2|zorin_danila_artemovich|0.1740|2|
|3823B1PE2|zyuzin_nikita|0.1740|7|
|**REF**|**REF**|**0.1740**|**-**|
|3823B1PE4|tsarkov_klim|0.1750|2|
|3823B1PE2|olesnitskiy_vladimir|0.1760|9|
|3823B1PE2|sinev_artem|0.1980|4|
|3823B1PE1|lazareva_ann|0.2180|9|
|3823B1PE1|kondrashova_victoria|0.2490|6|
|3823B1PE4|terekhov_dmitriy|0.3640|4|

## 8_gelu_ocl (134217728 elements)
|Group|Name|Result|Rank|
|-----|----|------|----|
|3823B1PE2|viderman_aleksandra|0.1932|5|
|3823B1FI1|dorofeev_ivan|0.1984|2|
|3823B1PE2|akhmetov_daniil|0.1986|7|
|3823B1PE3|ermakov_alexei|0.2011|4|
|3823B1PE2|zyuzin_nikita|0.2012|6|
|**FAST**|**FAST**|**0.2036**|**-**|
|3823B1FI1|zenin_anton|0.2050|1|
|3823B1PE4|karpich_ivan|0.2102|5|
|3823B1PE4|peryashkin_vasily|0.2377|3|
|3823B1PE3|dergachev_arseniy|0.2382|1|
|3823B1FI3|lifanov_kirill|0.2386|1|
|3823B1PE1|rusakova_aleksandra|0.2386|2|
|3823B1PE4|dilshodov_adkham|0.2399|1|
|3823B1PE2|vasiliev_mikhail|0.2410|4|
|3823B1PE1|otcheskov_semyon|0.2442|3|
|3823B1PE3|marin_lev|0.2457|3|
|3823B1PE2|klimenko_vladislav|0.2468|1|
|3823B1PE1|perepelkin_iaroslav|0.2472|4|
|3823B1PE1|ivanova_polina|0.2485|7|
|3823B1PE1|redkina_alina|0.2500|1|
|3823B1PE1|lazareva_ann|0.2509|8|
|3823B1PE2|kotelnikova_anastasia|0.2514|2|
|3823B1PE1|chernov_timur|0.2530|10|
|3823B1PE1|golovanov_daniil|0.2535|5|
|3823B1PE3|bortsova_angelina|0.2561|2|
|3823B1PE1|shilin_nikita|0.2573|9|
|3823B1PE4|terekhov_dmitriy|0.2737|4|
|3823B1PE4|tsarkov_klim|0.2764|2|
|**REF**|**REF**|**0.3289**|**-**|
|3823B1PE1|kondrashova_victoria|0.4422|6|
|3823B1PE2|zorin_danila_artemovich|0.4599|3|
|3823B1PE2|sinev_artem|TEST FAILED|-|
|3823B1PE2|olesnitskiy_vladimir|TEST FAILED|-|
|3823B1PE3|rastvorov_kirill|TEST FAILED|-|

# Tasks Done
## 3823B1FI1
|Group|Name|Passed|Score|
|-----|----|------|-----|
|3823B1FI1|dorofeev_ivan|**8/8**|**503**|
|3823B1FI1|kosolapov_vitaliy|1/8|60|
|3823B1FI1|krykov_egor|2/8|121|
|3823B1FI1|zenin_anton|**8/8**|**502**|

Passed: 2

## 3823B1FI2
|Group|Name|Passed|Score|
|-----|----|------|-----|
|3823B1FI2|ashihmin_daniil|1/8|59|
|3823B1FI2|chyokotov_alexey|4/8|244|
|3823B1FI2|rozenberg_alexey|3/8|185|
|3823B1FI2|sannikov_ivan|3/8|181|
|3823B1FI2|volkov_aleksei|3/8|183|

Passed: 0

## 3823B1FI3
|Group|Name|Passed|Score|
|-----|----|------|-----|
|3823B1FI3|lifanov_kirill|**8/8**|**512**|

Passed: 1

## 3823B1PE1
|Group|Name|Passed|Score|
|-----|----|------|-----|
|3823B1PE1|chernov_timur|**8/8**|**343**|
|3823B1PE1|golovanov_daniil|**8/8**|**390**|
|3823B1PE1|ivanova_polina|**8/8**|**404**|
|3823B1PE1|kondrashova_victoria|**8/8**|**379**|
|3823B1PE1|lazareva_ann|**8/8**|**370**|
|3823B1PE1|makoveeva_sofia|1/8|51|
|3823B1PE1|melnik_ivan|4/8|202|
|3823B1PE1|morozov_nikita|1/8|49|
|3823B1PE1|otcheskov_semyon|**8/8**|**464**|
|3823B1PE1|perepelkin_iaroslav|**8/8**|**462**|
|3823B1PE1|popova_elizaveta|6/8|242|
|3823B1PE1|redkina_alina|**8/8**|**472**|
|3823B1PE1|rusakova_aleksandra|**8/8**|**464**|
|3823B1PE1|shilin_nikita|**8/8**|**447**|
|3823B1PE1|tabalaev_anton|6/8|296|
|3823B1PE1|tsibareva_ekaterina|6/8|284|
|3823B1PE1|zhurin_ivan|5/8|233|
|3823B1PE1|zyazeva_svetlana|1/8|34|

Passed: 10

## 3823B1PE2
|Group|Name|Passed|Score|
|-----|----|------|-----|
|3823B1PE2|akhmetov_daniil|**8/8**|**436**|
|3823B1PE2|klimenko_vladislav|**8/8**|**453**|
|3823B1PE2|kolotukhin_alexander|4/8|218|
|3823B1PE2|kotelnikova_anastasia|**8/8**|**440**|
|3823B1PE2|olesnitskiy_vladimir|7/8|345|
|3823B1PE2|ovchinnikov_matvey|2/8|102|
|3823B1PE2|sinev_artem|7/8|389|
|3823B1PE2|tsyplakov_kirill|4/8|199|
|3823B1PE2|vasiliev_mikhail|**8/8**|**463**|
|3823B1PE2|viderman_aleksandra|**8/8**|**464**|
|3823B1PE2|zorin_danila_artemovich|**8/8**|**439**|
|3823B1PE2|zyuzin_nikita|**8/8**|**430**|

Passed: 7

## 3823B1PE3
|Group|Name|Passed|Score|
|-----|----|------|-----|
|3823B1PE3|afanasyev_alexandr|6/8|309|
|3823B1PE3|balchunayte_zlata|3/8|153|
|3823B1PE3|batkov_filipp|4/8|226|
|3823B1PE3|bortsova_angelina|**8/8**|**474**|
|3823B1PE3|borunov_vladislav|3/8|167|
|3823B1PE3|dergachev_arseniy|**8/8**|**493**|
|3823B1PE3|ermakov_alexei|**8/8**|**470**|
|3823B1PE3|marin_lev|**8/8**|**472**|
|3823B1PE3|rastvorov_kirill|7/8|406|

Passed: 4

## 3823B1PE4
|Group|Name|Passed|Score|
|-----|----|------|-----|
|3823B1PE4|baranov_andrey|3/8|158|
|3823B1PE4|dilshodov_adkham|**8/8**|**497**|
|3823B1PE4|karpich_ivan|**8/8**|**442**|
|3823B1PE4|peryashkin_vasily|**8/8**|**467**|
|3823B1PE4|remizov_kirill|6/8|355|
|3823B1PE4|romanov_michail|6/8|321|
|3823B1PE4|safronov_maksim|4/8|194|
|3823B1PE4|terekhov_dmitriy|**8/8**|**421**|
|3823B1PE4|trofimov_nikita|4/8|191|
|3823B1PE4|tsarkov_klim|**8/8**|**431**|
|3823B1PE4|urin_oleg|4/8|210|
|3823B1PE4|zaharov_gleb|6/8|339|

Passed: 5

**Total Passed: 29**

---
*Maximum Score: 512 (64 per task)*
