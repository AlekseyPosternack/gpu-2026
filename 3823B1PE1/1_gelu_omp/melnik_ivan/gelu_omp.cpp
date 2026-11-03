#include "gelu_omp.h"

#include <cmath>
#include <cstddef>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const std::size_t n = input.size();
    std::vector<float> result(n);

    if (n == 0) {
        return result;
    }

    const float* __restrict__ in = input.data();
    float* __restrict__ out = result.data();

    // kScale2 = -2.0 * sqrt(2/pi) * log2(e)
    constexpr float kScale2 = -2.302208198144325f;
    constexpr float kCoef = 0.044715f;

    const std::size_t n8 = n & ~std::size_t(7);

    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n8; i += 8) {

        #pragma omp simd
        for (std::size_t j = 0; j < 8; ++j) {
            const float x = in[i + j];
            
            // (x^2 * 0.044715) + 1.0
            const float poly = std::fmaf(kCoef, x * x, 1.0f);
            
            // -2 * sqrt(2/pi) * log2(e) * x * (1 + 0.044715 * x^2)
            const float arg = kScale2 * x * poly;

            out[i + j] = x / (1.0f + exp2f(arg));
        }
    }

    for (std::size_t i = n8; i < n; ++i) {
        const float x = in[i];
        const float poly = std::fmaf(kCoef, x * x, 1.0f);
        const float arg = kScale2 * x * poly;
        out[i] = x / (1.0f + exp2f(arg));
    }

    return result;
}