#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    size_t n = input.size();
    std::vector<float> result(n);

    const float sqrt_2_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        float x = input[i];
        float x3 = x * x * x;
        
        float arg = sqrt_2_pi * (x + coeff * x3);
        
        float exp_val = expf(2.0f * arg);
        float tanh_approx = (exp_val - 1.0f) / (exp_val + 1.0f);
        
        result[i] = 0.5f * x * (1.0f + tanh_approx);
    }

    return result;
}