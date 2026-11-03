#include "gelu_omp.h"

#include <cmath>
#include <cstddef>
#include <vector>

std::vector<float> GeluOMP(
    const std::vector<float>& input
) {
    std::vector<float> result(input.size());

    constexpr float k = 0.7978845608f;
    constexpr float c = 0.044715f;

    const std::ptrdiff_t n =
        static_cast<std::ptrdiff_t>(input.size());

    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        float x = input[i];

        float u = k * (x + c * x * x * x);

        result[i] = x / (1.0f + std::exp(-2.0f * u));
    }

    return result;
}