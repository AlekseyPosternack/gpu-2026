#include "gelu_omp.h"

#include <cmath>
#include <cstddef>

// GELU(x) = 0.5 * x * (1 + tanh(z)), z = sqrt(2/pi) * (x + 0.044715 * x^3)
// Using 0.5 * (1 + tanh(z)) == 1 / (1 + exp(-2z)) reduces to a single exp:
//     GELU(x) = x / (1 + exp(-2 * sqrt(2/pi) * (x + 0.044715 * x^3)))
std::vector<float> GeluOMP(const std::vector<float>& input) {
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    constexpr float kA = -2.0f * kSqrt2OverPi;
    constexpr float kB = kA * 0.044715f;

    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(input.size());
    std::vector<float> output(input.size());

    const float* __restrict in = input.data();
    float* __restrict out = output.data();

#pragma omp parallel for simd schedule(static)
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        const float x = in[i];
        const float t = x * (kA + kB * x * x);
        out[i] = x / (1.0f + std::exp(t));
    }

    return output;
}
