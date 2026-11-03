#ifndef SOFTMAX_CUDA_H
#define SOFTMAX_CUDA_H

#include <vector>

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count);

#endif // SOFTMAX_CUDA_H
