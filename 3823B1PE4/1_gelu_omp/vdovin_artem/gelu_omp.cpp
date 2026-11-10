#include "gelu_omp.h"

#include <cmath>
#include <cstddef>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const std::ptrdiff_t size = static_cast<std::ptrdiff_t>(input.size());
    std::vector<float> output(input.size());

    const float* in = input.data();
    float* out = output.data();

    const float k = 1.5957691216057308f;
    const float c = 0.044715f;

#pragma omp parallel for simd schedule(static)
    for (std::ptrdiff_t i = 0; i < size; ++i) {
        const float x = in[i];
        const float t = k * x * (1.0f + c * x * x);
        out[i] = t < -80.0f ? 0.0f : x / (1.0f + std::exp(-t));
    }

    return output;
}
