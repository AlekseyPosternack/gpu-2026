#include "gelu_omp.h"

#include <cmath>
#include <cstddef>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float> &input)
{

    const std::size_t n = input.size();
    std::vector<float> result(input.size());

#pragma omp parallel for
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        float x = input[i];

        float x3 = x * x * x;
        float value = std::sqrt(2.0f / 3.14159265358979323846f) * (x + 0.044715f * x3);

        result[i] = 0.5f * x * (1.0f + std::tanh(value));
    }

    return result;
}