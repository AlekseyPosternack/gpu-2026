#include "gelu_omp.h"

#include <cmath>
#include <cstddef>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> result(input.size());
    constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
    constexpr float kCubicCoefficient = 0.044715f;

    const std::size_t count = input.size();
#pragma omp parallel for schedule(static) if (count >= 4096)
    for (std::size_t i = 0; i < count; ++i) {
        const float x = input[i];
        const float argument = kSqrtTwoOverPi * (x + kCubicCoefficient * x * x * x);
        result[i] = 0.5f * x * (1.0f + std::tanh(argument));
    }

    return result;
}
