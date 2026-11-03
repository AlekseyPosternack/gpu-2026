#include "gelu_omp.h"
#include <cmath>
#include <cstddef>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(input.size());
    std::vector<float> output(input.size());
    const float* __restrict src = input.data();
    float* __restrict dst = output.data();
    constexpr float scale = -2.302208198144325f;

#pragma omp parallel for simd schedule(static) if(n >= 32768)
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        const float x = src[i];
        const float arg = scale * x * (1.0f + 0.044715f * x * x);
        dst[i] = x / (1.0f + std::exp2(arg));
    }
    return output;
}
