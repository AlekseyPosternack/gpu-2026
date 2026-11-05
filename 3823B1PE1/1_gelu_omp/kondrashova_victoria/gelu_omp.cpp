#include "gelu_omp.h"

#include <cmath>

namespace {
constexpr float kSqrt2OverPi = 0.7978845608028654f;
constexpr float kGeluCoef = 0.044715f;
}  // namespace

std::vector<float> GeluOMP(const std::vector<float>& input) {
    size_t size = input.size();
    std::vector<float> output(size);

    const float* in = input.data();
    float* out = output.data();

#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < size; i++) {
        float x = in[i];
        float x3 = x * x * x;
        float z = kSqrt2OverPi * (x + kGeluCoef * x3);
        out[i] = x / (1.0f + std::exp(-2.0f * z));
    }

    return output;
}
