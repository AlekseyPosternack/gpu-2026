#include "gelu_omp.h"
#include <cmath>
#include <cstddef>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    if (input.empty()) {
        return {};
    }

    const std::size_t n = input.size();
    std::vector<float> output(n);

    const float* in = input.data();
    float* out = output.data();

    constexpr float c1 = 1.5957691216057308f;
    constexpr float c2 = 0.0713548162726009f;

    const long long total = static_cast<long long>(n);

    #pragma omp parallel for simd schedule(static)
    for (long long i = 0; i < total; ++i) {
        const float x = in[i];
        out[i] = x / (1.0f + std::exp(-x * (c1 + c2 * x * x)));
    }

    return output;
}
