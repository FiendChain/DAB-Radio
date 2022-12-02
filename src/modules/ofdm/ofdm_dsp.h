#pragma once

#include <complex>
#include <stdint.h>
#include "utility/span.h"

constexpr int16_t FIXED_POINT_SCALING = 128;

// Helpers for doing vectorised dsp for OFDM demodulator
float apply_pll_scalar(
    tcb::span<const std::complex<int16_t>> x, 
    tcb::span<std::complex<int16_t>> y, 
    const float freq_offset,
    const float dt0=0.0f, 
    const int16_t scale=FIXED_POINT_SCALING);

float apply_pll_avx2(
    tcb::span<const std::complex<int16_t>> x, 
    tcb::span<std::complex<int16_t>> y, 
    const float freq_offset,
    const float dt0=0.0f,
    const int16_t scale=FIXED_POINT_SCALING);