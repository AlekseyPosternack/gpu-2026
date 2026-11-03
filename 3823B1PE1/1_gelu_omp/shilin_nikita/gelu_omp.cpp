#include "gelu_omp.h"

#include <cmath>
#include <cstddef>
#include <malloc.h>

namespace {

void reuse_large_allocs() {
    mallopt(M_MMAP_THRESHOLD, 1 << 30);
    mallopt(M_TRIM_THRESHOLD, -1);
}

struct RawVec : std::vector<float> {
    void grow(std::size_t n) {
        reserve(n);
        _M_impl._M_finish = _M_impl._M_start + n;
    }
};

__attribute__((target("avx2,fma")))
void gelu_kernel(const float* __restrict__ in, float* __restrict__ out, std::ptrdiff_t n) {
    constexpr float kA = 2.302208198144325f;
    constexpr float kB = 0.1029432395800235f;
#pragma omp parallel for simd schedule(static) aligned(in, out : 16)
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        const float x = in[i];
        const float t = x * (kA + kB * x * x);
        out[i] = x / (1.0f + exp2f(-t));
    }
}

}

std::vector<float> GeluOMP(const std::vector<float>& input) {
    static const int kOnce = (reuse_large_allocs(), 0);
    (void)kOnce;

    const std::size_t n = input.size();
    RawVec raw;
    if (n == 0) {
        return raw;
    }

    raw.grow(n);
    gelu_kernel(input.data(), raw.data(), static_cast<std::ptrdiff_t>(n));

    std::vector<float> output;
    output.swap(raw);
    return output;
}
