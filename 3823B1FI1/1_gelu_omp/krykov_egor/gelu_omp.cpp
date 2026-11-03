#include "gelu_omp.h"
#include <cmath>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const int n = static_cast<int>(input.size());
    std::vector<float> result(n);

    const float* in = input.data();
    float* out = result.data();

    const float k = 2.0f * 0.7978845608028654f;
    const float c = 0.044715f;

    #pragma omp parallel for simd
    for (int i = 0; i < n; ++i) {
        const float x = in[i];
        const float z = k * (x + c * x * x * x);
        out[i] = x - x / (std::exp(z) + 1.0f);
    }

    return result;
}