#include "gelu_omp.h"

#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const int size = static_cast<int>(input.size());
    std::vector<float> output(input.size());

    const float* in = input.data();
    float* out = output.data();

    const float k0 = -2.0f * 0.7978845608028654f;  // -2 * sqrt(2 / pi)
    const float k1 = k0 * 0.044715f;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < size; ++i) {
        const float x = in[i];
        out[i] = x / (1.0f + std::exp(x * (k0 + k1 * x * x)));
    }

    return output;
}
