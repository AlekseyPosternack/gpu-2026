#include "gelu_omp.h"

#include <cmath>
#include <cstddef>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const std::size_t n = input.size();
    std::vector<float> result(n);

    const float* in = input.data();
    float* out = result.data();

    const float c1 = -1.5957691216057308f;
    const float c2 = -1.5957691216057308f * 0.044715f;

    const long long size = static_cast<long long>(n);

    #pragma omp parallel for simd schedule(static)
    for (long long i = 0; i < size; ++i) {
        const float x = in[i];
        const float x2 = x * x;
        const float t = x * (c1 + c2 * x2);
        out[i] = x / (1.0f + std::exp(t));
    }

    return result;
}