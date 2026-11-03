#include "gelu_omp.h"

#include <cmath>
#include <cstddef>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> output(input.size());

    #pragma omp parallel for
    for (std::ptrdiff_t i = 0;
         i < static_cast<std::ptrdiff_t>(input.size());
         ++i) {
        const float x = input[static_cast<std::size_t>(i)];
        const float x3 = x * x * x;
        const float inner =
            0.7978845608f * (x + 0.044715f * x3);

        output[static_cast<std::size_t>(i)] =
            0.5f * x * (1.0f + std::tanh(inner));
    }

    return output;
}
