#include "gelu_omp.h"
#include <vector>
#include <cmath>
#include <cstddef>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t size = input.size();
    std::vector<float> output(size);

    const float* in_ptr = input.data();
    float* out_ptr = output.data();

    const float C1 = -1.59576912f;
    const float C3 = -0.07135481f;

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < size; ++i) {
        float x = in_ptr[i];
        
        float exponent = x * (C1 + C3 * x * x);
        
        out_ptr[i] = x / (1.0f + std::exp(exponent));
    }

    return output;
}