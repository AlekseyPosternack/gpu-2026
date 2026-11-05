#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t n = input.size();
    std::vector<float> output(n);

    const float* in = input.data();
    float* out = output.data();

    const float kSqrt2OverPi = 0.7978845608028654f; 
    const float kCoeff       = 0.044715f;

    
    #pragma omp parallel for simd schedule(static) \
        default(none) shared(in, out, n, kSqrt2OverPi, kCoeff)
    for (size_t i = 0; i < n; ++i) {
        const float x  = in[i];
        const float x3 = x * x * x;
        const float z  = kSqrt2OverPi * (x + kCoeff * x3);
        const float e = expf(-2.0f * z);
        out[i] = x / (1.0f + e);
    }

    return output;
}