#include "gelu_omp.h"

#include <cmath>
#include <cstddef>

// 0.5 * x * (1 + tanh(z)) == x / (1 + exp(-2z)),
// where z = sqrt(2 / pi) * (x + 0.044715 * x^3)
std::vector<float> GeluOMP(const std::vector<float>& input) {
    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(input.size());
    std::vector<float> output(input.size());

    const float* __restrict__ in = input.data();
    float* __restrict__ out = output.data();

    constexpr float c1 = -1.5957691216057308f;  // -2 * sqrt(2 / pi)
    constexpr float c2 = c1 * 0.044715f;

#pragma omp parallel for simd schedule(static)
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        const float x = in[i];
        const float t = x * (c1 + c2 * x * x);
        out[i] = x / (1.0f + std::exp(t));
    }

    return output;
}
