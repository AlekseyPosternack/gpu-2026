#include "gelu_omp.h"

#include <cmath>
#include <cstddef>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    const std::ptrdiff_t size = static_cast<std::ptrdiff_t>(input.size());
    constexpr float sqrt_two_over_pi = 0.7978845608028654f;
    constexpr float coefficient = 0.044715f;

#pragma omp parallel for schedule(static) if(size >= 4096)
    for (std::ptrdiff_t i = 0; i < size; ++i) {
        const float x = input[i];
        const float argument = sqrt_two_over_pi * (x + coefficient * x * x * x);
        output[i] = 0.5f * x * (1.0f + std::tanh(argument));
    }

    return output;
}