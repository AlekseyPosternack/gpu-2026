#include "gelu_omp.h"
#include <cmath>
#include <vector>

#pragma GCC target("avx2,fma")
#pragma GCC optimize("O3,unroll-loops,fast-math")

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t n = input.size();
    std::vector<float> result(n);
    
    const float* __restrict__ in = input.data();
    float* __restrict__ out = result.data();

    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    const float half = 0.5f;

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const float x = in[i];
        const float x_cubed = x * x * x;
        const float inner = sqrt_2_over_pi * (x + coeff * x_cubed);

        out[i] = half * x * (1.0f + std::tanh(inner));
    }

    return result;
}