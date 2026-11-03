#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> output(input.size());

    const float sqrt_2_over_pi = 0.7978845608f;
    const float coef = 0.044715f;

    #pragma omp parallel for simd
    for (size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        float inner = sqrt_2_over_pi * (x + coef * x * x * x);
        float e = expf(2.0f * inner);
        float tanh_val = (e - 1.0f) / (e + 1.0f);
        output[i] = 0.5f * x * (1.0f + tanh_val);
    }
    return output;
}