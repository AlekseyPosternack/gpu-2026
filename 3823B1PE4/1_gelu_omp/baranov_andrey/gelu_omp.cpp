#include "gelu_omp.h"

#include <cmath>
#include <cstddef>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const std::size_t size = input.size();
    std::vector<float> result(size);
    constexpr float sqrt_two_over_pi = 0.7978845608028654f;

#pragma omp parallel for simd schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(size); ++i) {
        const float x = input[i];
        const float argument = sqrt_two_over_pi * (x + 0.044715f * x * x * x);
        result[i] = 0.5f * x * (1.0f + std::tanh(argument));
    }

    return result;
}