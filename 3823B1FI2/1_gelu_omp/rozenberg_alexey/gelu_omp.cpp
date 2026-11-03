#include "gelu_omp.h"
#include <cmath>
#include <cstddef>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const std::size_t n = input.size();
    std::vector<float> output(n);

    const float* in = input.data();
    float* out = output.data();
    
    // 2 * sqrt(2/pi)
    const float c1 = 1.5957691216f;
    // 2 * sqrt(2/pi) * 0.044715
    const float c2 = 0.0713548163f;

    const long long total = static_cast<long long>(n);

    #pragma omp parallel for simd schedule(static)
    for (long long i = 0; i < total; ++i) {
        const float x = in[i];
        const float x2 = x * x;
        // GELU(x) = x / (1 + exp(-2*sqrt(2/pi)*(x + 0.044715*x^3)))
        out[i] = x / (1.0f + std::exp(-x * (c1 + c2 * x2)));
    }

    return output;
}