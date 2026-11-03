#include "gelu_omp.h"

#include <cmath>
#include <cstddef>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> result(input.size());

    const float k = 1.5957691216f;

    #pragma omp parallel for
    for (std::size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        float x3 = x * x * x;

        float z = -k * (x + 0.044715f * x3);

        result[i] = x / (1.0f + std::exp(z));
    }

    return result;
}
