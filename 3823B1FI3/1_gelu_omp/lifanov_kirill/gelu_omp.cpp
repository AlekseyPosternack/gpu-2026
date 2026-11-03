#include "gelu_omp.h"

#include <cmath>
#include <cstddef>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const std::size_t n = input.size();
    std::vector<float> output(n);

    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    constexpr float kCoef = 0.044715f;

    constexpr float kNegTwoSqrt = -2.0f * kSqrt2OverPi;

    const float* __restrict in  = input.data();
    float* __restrict out = output.data();

    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
        const float x = in[i];
        const float x3 = x * x * x;

        const float arg = kNegTwoSqrt * (x + kCoef * x3);

        out[i] = x / (1.0f + std::exp(arg));
    }

    return output;
}