#include "gelu_omp.h"

#include <cmath>
#include <cstddef>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> result(input.size());
    constexpr float scale = 1.5957691216057308f;

    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < input.size(); ++i) {
        const float x = input[i];
        const float exponent = scale * x * (1.0f + 0.044715f * x * x);
        result[i] = x / (1.0f + std::exp(-exponent));
    }

    return result;
}
