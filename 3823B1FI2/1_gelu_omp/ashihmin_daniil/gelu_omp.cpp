#include "gelu_omp.h"
#include <cmath>
#include <cstdint>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const auto size = input.size();
    if (size == 0) {
        return {};
    }

    std::vector<float> result(size);

    const float* src = input.data();
    float* dst = result.data();

    //A = 2 * sqrt(2 / pi)
    //B = 2 * sqrt(2 / pi) * 0.044715
    constexpr float A = 1.5957691216057307f;
    constexpr float B = 0.0713548162726009f;

    const auto total_elements = static_cast<int64_t>(size);

    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < total_elements; ++i) {
        const float val = src[i];
        const float exp_arg = -val * (A + B * val * val);
        dst[i] = val / (1.0f + std::exp(exp_arg));
    }

    return result;
}