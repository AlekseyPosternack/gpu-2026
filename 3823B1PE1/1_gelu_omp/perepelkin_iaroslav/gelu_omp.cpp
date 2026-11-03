#include <cmath>

#include "gelu_omp.h"


namespace {
// GELU(x) = x / (1 + exp(-x * (kA + kB * x^2)))
constexpr float kLog2e = 1.44269504089f;
constexpr float kA = 1.59576912f * kLog2e;
constexpr float kB = 0.0713548f * kLog2e;
}

std::vector<float> GeluOMP(const std::vector<float>& input) {
    if (input.empty()) {
        return {};
    }

    const size_t size = input.size();
    std::vector<float> result(size);

    const float* __restrict__ in_ptr = input.data();
    float* __restrict__ out_ptr = result.data();

    #pragma omp parallel for simd aligned(in_ptr, out_ptr: 16) schedule(static) default(none) shared(size, in_ptr, out_ptr)
    for (size_t i = 0; i < size; ++i) {
        float x = in_ptr[i];
        float t = x * (kA + kB * x * x);
        out_ptr[i] = x / (1.0f + std::exp2f(-t));
    }

    return result;
}
