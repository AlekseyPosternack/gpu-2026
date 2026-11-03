#ifndef GEMM_CUBLAS_H
#define GEMM_CUBLAS_H

#include <vector>

std::vector<float> GemmCUBLAS(const std::vector<float>& left_matrix, const std::vector<float>& right_matrix, int matrix_size);

#endif
