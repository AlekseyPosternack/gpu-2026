#include "gelu_omp.h"

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>

namespace {

constexpr float kK1 = 1.5957691216057308f;
constexpr float kK2 = 0.0713548162726009f;
constexpr float kExpHi = 88.0f;
constexpr float kExpLo = -87.3f;
constexpr float kLog2e = 1.44269504088896341f;
constexpr float kLn2Hi = 0.693359375f;
constexpr float kLn2Lo = -2.12194440e-4f;
constexpr float kRound = 12582912.0f;
constexpr uint32_t kRoundBits = 0x4b400000u;
constexpr uint32_t kSignMask = 0x80000000u;
constexpr size_t kChunk = 1 << 16;

inline uint32_t AsBits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

inline float AsFloat(uint32_t u) {
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

inline float Gelu(float x) {
    const float t = -x * (kK1 + kK2 * x * x);

    const uint32_t n = AsBits(t * kLog2e + kRound) - kRoundBits;
    const float fn = static_cast<float>(static_cast<int32_t>(n));
    const float r = (t - fn * kLn2Hi) - fn * kLn2Lo;

    float p = 1.9875691500e-4f;
    p = p * r + 1.3981999507e-3f;
    p = p * r + 8.3334519073e-3f;
    p = p * r + 4.1665795894e-2f;
    p = p * r + 1.6666665459e-1f;
    p = p * r + 5.0000001201e-1f;
    const float e = (p * r * r + r + 1.0f) * AsFloat((n + 127u) << 23);

    const uint32_t hi = 0u - static_cast<uint32_t>(t > kExpHi);
    const uint32_t lo = 0u - static_cast<uint32_t>(t < kExpLo);
    const uint32_t xb = AsBits(x);
    const uint32_t y = AsBits(x / (1.0f + e));
    return AsFloat((y & ~(hi | lo)) | (xb & kSignMask & hi) | (xb & lo));
}

void GeluRange(const float* in, float* out, size_t n) {
#pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        out[i] = Gelu(in[i]);
    }
}

}

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t n = input.size();
    const size_t chunks = (n + kChunk - 1) / kChunk;

    std::vector<float> output;
    output.reserve(n);
    const float* in = input.data();
    float* out = output.data();

    std::atomic<size_t> ready{0};
    std::atomic<size_t> next{0};

#pragma omp parallel
    {
        if (omp_get_thread_num() == 0) {
            for (size_t c = 1; c <= chunks; ++c) {
                output.resize(std::min(n, c * kChunk));
                ready.store(c, std::memory_order_release);
            }
        }

        for (size_t c = next++; c < chunks; c = next++) {
            while (ready.load(std::memory_order_acquire) <= c) {
                std::this_thread::yield();
            }
            const size_t begin = c * kChunk;
            GeluRange(in + begin, out + begin, std::min(kChunk, n - begin));
        }
    }

    return output;
}
