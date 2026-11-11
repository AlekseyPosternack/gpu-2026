#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t n = input.size();
    std::vector<float> output(n);
    
    const float sqrt_2_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    
    #pragma omp parallel for simd
    for (size_t i = 0; i < n; ++i) {
        float x = input[i];
        float y = sqrt_2_pi * x * (1.0f + coeff * x * x);
        output[i] = x / (1.0f + expf(-2.0f * y));
    }
    
    return output;
}

// GELU(x) = 0.5 · x · (1 + tanh(y))
// y = sqrt(2/pi) · (x + 0.044715 · x^3) = sqrt(2/pi) · x · (1 + 0.044715 · x^2)

// tanh(y) = exp(-2y)
// GELU(x) = x / (1 + exp(-2y))
