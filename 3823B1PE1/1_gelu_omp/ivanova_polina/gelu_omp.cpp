#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t n = input.size();
    if (n == 0) return {};

    // Выделяем память без предварительной инициализации нулями
    std::vector<float> output;
    output.reserve(n);
    output.resize(n);

    const float* __restrict__ in = input.data();
    float* __restrict__ out = output.data();

    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;

    const size_t n_aligned = n & ~7ULL; // Кратно 8 для полных 256-битных AVX2 регистров

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n_aligned; i += 8) {
        // Разворот на 8 элементов для полного задействования AVX2
        float x0 = in[i];     float x1 = in[i + 1]; float x2 = in[i + 2]; float x3 = in[i + 3];
        float x4 = in[i + 4]; float x5 = in[i + 5]; float x6 = in[i + 6]; float x7 = in[i + 7];

        float in0 = sqrt_2_over_pi * (x0 + coeff * (x0 * x0 * x0));
        float in1 = sqrt_2_over_pi * (x1 + coeff * (x1 * x1 * x1));
        float in2 = sqrt_2_over_pi * (x2 + coeff * (x2 * x2 * x2));
        float in3 = sqrt_2_over_pi * (x3 + coeff * (x3 * x3 * x3));
        float in4 = sqrt_2_over_pi * (x4 + coeff * (x4 * x4 * x4));
        float in5 = sqrt_2_over_pi * (x5 + coeff * (x5 * x5 * x5));
        float in6 = sqrt_2_over_pi * (x6 + coeff * (x6 * x6 * x6));
        float in7 = sqrt_2_over_pi * (x7 + coeff * (x7 * x7 * x7));

        float e0 = std::exp(2.0f * in0); float e1 = std::exp(2.0f * in1);
        float e2 = std::exp(2.0f * in2); float e3 = std::exp(2.0f * in3);
        float e4 = std::exp(2.0f * in4); float e5 = std::exp(2.0f * in5);
        float e6 = std::exp(2.0f * in6); float e7 = std::exp(2.0f * in7);

        out[i]     = 0.5f * x0 * (1.0f + (e0 - 1.0f) / (e0 + 1.0f));
        out[i + 1] = 0.5f * x1 * (1.0f + (e1 - 1.0f) / (e1 + 1.0f));
        out[i + 2] = 0.5f * x2 * (1.0f + (e2 - 1.0f) / (e2 + 1.0f));
        out[i + 3] = 0.5f * x3 * (1.0f + (e3 - 1.0f) / (e3 + 1.0f));
        out[i + 4] = 0.5f * x4 * (1.0f + (e4 - 1.0f) / (e4 + 1.0f));
        out[i + 5] = 0.5f * x5 * (1.0f + (e5 - 1.0f) / (e5 + 1.0f));
        out[i + 6] = 0.5f * x6 * (1.0f + (e6 - 1.0f) / (e6 + 1.0f));
        out[i + 7] = 0.5f * x7 * (1.0f + (e7 - 1.0f) / (e7 + 1.0f));
    }

    // Хвост массива
    #pragma omp parallel for schedule(static)
    for (size_t i = n_aligned; i < n; ++i) {
        float x = in[i];
        float inner = sqrt_2_over_pi * (x + coeff * (x * x * x));
        float e = std::exp(2.0f * inner);
        out[i] = 0.5f * x * (1.0f + (e - 1.0f) / (e + 1.0f));
    }

    return output;
}