#include "gelu_omp.h"

#include <cmath>
#include <cstddef>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    // 2 * sqrt(2 / pi) and 2 * sqrt(2 / pi) * 0.044715, respectively.
    // This is algebraically equivalent to the tanh-based formula:
    // 0.5 * x * (1 + tanh(a)) == x / (1 + exp(-2 * a)).
    constexpr float kLinear = -1.5957691216057308f;
    constexpr float kCubic = -0.0713548162726009f;

    const std::size_t size = input.size();
    std::vector<float> output(size);

    const float* __restrict in = input.data();
    float* __restrict out = output.data();

#pragma omp parallel for simd schedule(static)
    for (std::ptrdiff_t i = 0;
         i < static_cast<std::ptrdiff_t>(size);
         ++i) {
        const float x = in[i];
        const float xSquared = x * x;
        out[i] = x / (1.0f + std::exp(x * (kLinear + kCubic * xSquared)));
    }
    return output;
}
