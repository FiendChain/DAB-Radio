
#define _USE_MATH_DEFINES
#include <cmath>

#include "ofdm_dsp.h"
#include <complex>
#include "utility/span.h"

#include <immintrin.h>

const float Ts = 1.0f/2.048e6f;

float apply_pll_scalar(
    tcb::span<const std::complex<int16_t>> x, 
    tcb::span<std::complex<int16_t>> y, 
    const float freq_offset,
    const float dt0, 
    const int16_t scale) 
{
    const size_t N = x.size();
    const float dt_step = 2.0f * (float)M_PI * freq_offset * Ts;
    const bool is_large_offset = std::abs(freq_offset) > 1500.0f;

    float dt = dt0;
    for (int i = 0; i < N; i++) {
        const auto pll = std::complex<int16_t>(
            (int16_t)(std::cos(dt) * (float)FIXED_POINT_SCALING),
            (int16_t)(std::sin(dt) * (float)FIXED_POINT_SCALING));

        y[i] = x[i] * pll;
        dt += dt_step;
        if (is_large_offset) {
            dt = std::fmod(dt, 2.0f*(float)M_PI);
        }
    }

    for (int i = 0; i < N; i++) {
        y[i] /= scale;
    }

    return dt;
}

float apply_pll_avx2(
    tcb::span<const std::complex<int16_t>> x, 
    tcb::span<std::complex<int16_t>> y, 
    const float freq_offset,
    const float dt0, 
    const int16_t scale) 
{
    const auto N = x.size();
    const float dt_step = 2.0f * (float)M_PI * freq_offset * Ts;
    const bool is_large_offset = std::abs(freq_offset) > 1500.0f;

    // 256bits = 32bytes = 8*4bytes
    const int K = 8;
    const auto M = N/K;

    const __m256i real_mask = _mm256_set1_epi32(0x0000FFFF);
    const __m256i imag_mask = _mm256_set1_epi32(0xFFFF0000);

    __m256i 
        a0, a1,
        b0, b1, b2, b3, b4, 
        real_res, imag_res;

    __m256i x1_pack;

    __m256i x0_pack;
    __m256i y_pack;

    // auto* x0_pack = reinterpret_cast<const __m256i*>(x0.data());
    // auto* y_pack = reinterpret_cast<__m256i*>(y.data());

    // 4bytes per float
    // 4bytes = 2*2bytes per complex

    __m256 dt_pack;
    __m256 dt_step_pack;
    const float dt_step_pack_stride = dt_step * K;
    {
        float x = 0.0f;
        for (int i = 0; i < K; i++) {
            dt_step_pack.m256_f32[i] = x;
            x += dt_step;
        }
    }
    __m256 cos_res, sin_res;
    const auto pll_magnitude = _mm256_set1_ps((float)FIXED_POINT_SCALING);
    const auto output_rescale = _mm256_set1_epi16(scale);

    float dt = dt0;
    for (int i = 0; i < M; i++) {
        memcpy(x0_pack.m256i_i16, &x[i*K], sizeof(std::complex<int16_t>)*K);

        dt_pack = _mm256_set1_ps(dt);
        dt_pack = _mm256_add_ps(dt_pack, dt_step_pack);
        for (int j = 0; j < K; j++) {
            dt +=  dt_step;
        }
        // dt += dt_step_pack_stride;
        if (is_large_offset) {
            dt = std::fmod(dt, 2.0f*(float)M_PI);
        }

        sin_res = _mm256_sincos_ps(&cos_res, dt_pack);
        cos_res = _mm256_mul_ps(cos_res, pll_magnitude);
        sin_res = _mm256_mul_ps(sin_res, pll_magnitude);

        for (int j = 0; j < K; j++) {
            x1_pack.m256i_i16[2*j+0] = (int16_t)cos_res.m256_f32[j];
            x1_pack.m256i_i16[2*j+1] = (int16_t)sin_res.m256_f32[j];
        }

        // [ac bd]
        a0 = _mm256_mullo_epi16(x0_pack, x1_pack);
        // [bd ..]
        a1 = _mm256_bsrli_epi128(a0, 2);
        // [ac-bd 0]
        real_res = _mm256_sub_epi16(a0, a1);
        real_res = _mm256_and_si256(real_res, real_mask);

        // [d 0]
        b0 = _mm256_bsrli_epi128(x1_pack, 2);
        b0 = _mm256_and_si256(b0, real_mask);
        // [0 c]
        b1 = _mm256_bslli_epi128(x1_pack, 2);
        b1 = _mm256_and_si256(b1, imag_mask);
        // [d c]
        b2 = _mm256_or_si256(b0, b1);

        // [ad bc]
        b3 = _mm256_mullo_epi16(x0_pack, b2);
        // [.. ad]
        b4 = _mm256_bslli_epi128(b3, 2);
        // [0 bc+ad]
        imag_res = _mm256_add_epi16(b3, b4);
        imag_res = _mm256_and_si256(imag_res, imag_mask);

        y_pack = _mm256_or_si256(real_res, imag_res);
        y_pack = _mm256_div_epi16(y_pack, output_rescale);
        memcpy(&y[i*K], y_pack.m256i_i16, sizeof(std::complex<int16_t>)*K);
    }

    return dt0;
}