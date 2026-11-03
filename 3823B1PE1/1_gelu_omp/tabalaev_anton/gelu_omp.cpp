#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float> &input)
{
   const size_t n = input.size();
   if (n == 0)
   {
      return {};
   }

   std::vector<float> output(n);
   const float *__restrict__ in_ptr = input.data();
   float *__restrict__ out_ptr = output.data();

   constexpr float kBeta = -1.5957691216057308f; // -2.0f * sqrt(2.0f / pi)
   constexpr float kAlpha = 0.044715f;

#pragma omp parallel for simd schedule(static)
   for (size_t i = 0; i < n; ++i)
   {
      float x = in_ptr[i];
      out_ptr[i] = x / (1.0f + std::exp(kBeta * x * (1.0f + kAlpha * x * x)));
   }

   return output;
}
