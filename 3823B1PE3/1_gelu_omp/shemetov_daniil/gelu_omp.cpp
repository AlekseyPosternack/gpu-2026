#include "gelu_omp.h"

#include <cmath>
#include <cstddef>

std::vector<float> GeluOMP(const std::vector<float>& input) {
  const std::size_t size = input.size();
  if (size == 0) {
    return {};
  }

  std::vector<float> output(size);
  const float* in = input.data();
  float* out = output.data();

  constexpr float kMinusTwoSqrt2OverPi = -1.5957691216057307f;
  constexpr float kCubicCoeff = kMinusTwoSqrt2OverPi * 0.044715f;

  const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(size);

  #pragma omp parallel for simd schedule(static)
  for (std::ptrdiff_t i = 0; i < n; ++i) {
    const float x = in[i];
    const float arg = x * (kMinusTwoSqrt2OverPi + kCubicCoeff * x * x);
    
    out[i] = x / (1.0f + std::exp(arg));
  }

  return output;
}
