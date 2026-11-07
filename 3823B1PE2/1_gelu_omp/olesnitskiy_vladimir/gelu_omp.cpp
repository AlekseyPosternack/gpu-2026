#include "gelu_omp.h"

#include <cmath>
#include <cstddef>

#include <omp.h>

// 0.5 * x * (1 + tanh(z)) == x / (1 + exp(-2z)), z = sqrt(2/pi) * (x + 0.044715 * x^3)
std::vector<float> GeluOMP(const std::vector<float>& input) {
    const std::size_t size = input.size();
    std::vector<float> output(size);

    const float k1 = -2.0f * 0.7978845608028654f;
    const float k2 = k1 * 0.044715f;

    const float* in = input.data();
    float* out = output.data();

#pragma omp parallel for simd schedule(static)
    for (std::size_t i = 0; i < size; ++i) {
        const float x = in[i];
        out[i] = x / (1.0f + std::exp(x * (k1 + k2 * x * x)));
    }

    return output;
}
