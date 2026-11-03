
#include "gelu_omp.h"

#include <cmath>
#include <cstddef>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const std::size_t n = input.size();
    std::vector<float> result(n);

    const float a = 0.7978845608028654f;
    const float b = 0.044715f;

    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        const float x = input[i];
        const float x3 = x * x * x;

        const float t = a * (x + b * x3);

        result[i] = 0.5f * x * (1.0f + std::tanh(t));
    }

    return result;
}