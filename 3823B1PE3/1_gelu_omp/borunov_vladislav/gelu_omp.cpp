#include "gelu_omp.h"

#include <cmath>
#include <cstddef>


static constexpr float kScale = 1.5957691216057308f;
static constexpr float kCoeff = 0.044715f;

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(input.size());

    std::vector<float> result(n);

    const float* __restrict in  = input.data();
    float* __restrict out = result.data();


#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < n; i += 16) {
        out[i] = 0.0f;
    }

#pragma omp parallel for simd schedule(static) simdlen(8)
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        const float x  = in[i];
        const float x2 = x * x;
        const float z  = kScale * (x + kCoeff * x2 * x);
        out[i] = x / (1.0f + std::exp(-z));
    }

    return result;
}
