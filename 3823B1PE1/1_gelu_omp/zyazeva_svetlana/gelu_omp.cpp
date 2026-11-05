#include "gelu_omp.h"
#include <vector>
#include <cmath> 
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t size = input.size();
    std::vector<float> output(size);
    
    const float COEFF = 0.7978845608f;
    const float ALPHA = 0.044715f;
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; ++i) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = COEFF * (x + ALPHA * x3);
        float exp_2inner = expf(2.0f * inner);
        float tanh_val = (exp_2inner - 1.0f) / (exp_2inner + 1.0f);
        
        output[i] = 0.5f * x * (1.0f + tanh_val);
    }
    
    return output;
}
