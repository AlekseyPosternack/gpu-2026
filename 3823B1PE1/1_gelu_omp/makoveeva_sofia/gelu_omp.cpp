#include "gelu_omp.h"

#include <sys/mman.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#pragma GCC target("avx2,fma")

#pragma GCC optimize("no-trapping-math")

namespace {

constexpr float kC0 = 1.5957691216057308f;
constexpr float kC1 = kC0 * 0.044715f;    

inline float ExpMinus(float a) {
    const std::int32_t n = static_cast<std::int32_t>(a * 1.44269504f + 0.5f);
    const float r = (n * 0.693359375f - a) + n * -2.12194440e-4f;

    float p = 1.9875691500e-4f;
    p = p * r + 1.3981999507e-3f;
    p = p * r + 8.3334519073e-3f;
    p = p * r + 4.1665795894e-2f;
    p = p * r + 1.6666665459e-1f;
    p = p * r + 5.0000001201e-1f;
    const float exp_r = p * r * r + r + 1.0f;

    const std::int32_t bits = (127 - n) << 23;
    float pow2;
    std::memcpy(&pow2, &bits, sizeof(pow2));
    return exp_r * pow2;
}

inline float Gelu(float x) {
    const float two_u = x * (kC0 + kC1 * x * x);
    float a = two_u < 0.0f ? -two_u : two_u;
    a = a < 87.0f ? a : 87.0f;
    const float e = ExpMinus(a);
    const float numerator = x < 0.0f ? x * e : x;
    return numerator / (1.0f + e);
}

void UseHugePages(void* ptr, std::size_t bytes) {
    constexpr std::uintptr_t kHuge = std::uintptr_t{2} << 20;
    const auto addr = reinterpret_cast<std::uintptr_t>(ptr);
    const std::uintptr_t begin = (addr + kHuge - 1) & ~(kHuge - 1);
    const std::uintptr_t end = (addr + bytes) & ~(kHuge - 1);
    if (end > begin) {
        madvise(reinterpret_cast<void*>(begin), end - begin, MADV_HUGEPAGE);
    }
}

}  // namespace

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const std::int64_t n = static_cast<std::int64_t>(input.size());
    const float* in = input.data();

    std::vector<float> output;
    output.reserve(n);
    float* out = output.data();
    UseHugePages(out, n * sizeof(float));
#pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < n; i += 1024) {
        out[i] = 0.0f;
    }
    output.resize(n);

#pragma omp parallel for simd schedule(static)
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = Gelu(in[i]);
    }

    return output;
}
