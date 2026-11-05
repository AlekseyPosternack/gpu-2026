#ifndef __SOFTMAX_CUDA_H
#define __SOFTMAX_CUDA_H

#include <vector>

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count);

#endif // __SOFTMAX_CUDA_H
