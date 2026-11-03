#include "gelu_omp.h"

#include <cmath>
#include <vector>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    if (input.empty()) {
        return output;
    }

    constexpr float alpha = 0.7978845608f;
    constexpr float beta = 0.044715f;

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < input.size(); ++i) {
        const float x = input[i];
        const float x3 = x * x * x;
        const float tanh_arg = alpha * (x + beta * x3);
        output[i] = 0.5f * x * (1.0f + std::tanh(tanh_arg));
    }

    return output;
}
