#include "gelu_omp.h"

#include <cmath>
#include <cstddef>
#include <omp.h>

namespace {
    constexpr float SQRT_2_OVER_PI = 0.7978845608f;
    constexpr float CUBIC_COEFFICIENT = 0.044715f;
    constexpr float LOG2E = 1.44269504089f;
}

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const std::size_t n = input.size();

    std::vector<float> output(n);

    if (n == 0) {
        return output;
    }

    const float* in = input.data();
    float* out = output.data();

    #pragma omp parallel for simd schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        const float x = in[i];

        const float x2 = x * x;
        const float x3 = x2 * x;

        const float z = SQRT_2_OVER_PI *
                        (x + CUBIC_COEFFICIENT * x3);

        const float exponent = -2.0f * z * LOG2E;

        out[i] = x / (1.0f + ::exp2f(exponent));
    }

    return output;
}