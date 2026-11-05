#include <cmath>
#include <cstddef>
#include "gelu_omp.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {


inline float fast_tanh(float z) noexcept {
    const float e = std::exp(2.0f * z);
    return 1.0f - 2.0f / (e + 1.0f);
}


inline float gelu_scalar(float x) noexcept {
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    constexpr float kAlpha        = 0.044715f;
    const float x3 = x * x * x;
    return 0.5f * x * (1.0f + fast_tanh(kSqrt2OverPi * (x + kAlpha * x3)));
}

}

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const std::size_t n = input.size();
    std::vector<float> output(n);

    const float* __restrict in  = input.data();
    float*       __restrict out = output.data();

    #pragma omp parallel for schedule(static) if(n > 4096)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
        out[i] = gelu_scalar(in[i]);
    }

   return output;
}





