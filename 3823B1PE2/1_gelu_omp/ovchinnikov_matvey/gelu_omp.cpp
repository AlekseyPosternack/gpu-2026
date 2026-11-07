#include "gelu_omp.h"
#include <cmath>
std::vector<float> GeluOMP(const std::vector<float> &input) {
  size_t s = input.size();
  std::vector<float> res(s);
  const float k = -2 * sqrt(2 / M_PI);
#pragma omp parallel for simd
  for (size_t i = 0; i < s; ++i) {
    float x = input[i];
    res[i] = (x / (1.0f + std::exp(k * (x + 0.044715f * x * x * x))));
  }
  return res;
}