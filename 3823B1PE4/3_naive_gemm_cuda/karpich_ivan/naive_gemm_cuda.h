#ifndef KARPICH_IVAN_NAIVE_GEMM_CUDA_H
#define KARPICH_IVAN_NAIVE_GEMM_CUDA_H
#include <vector>

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n);

#endif
