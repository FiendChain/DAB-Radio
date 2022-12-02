#pragma once

#include <complex>
#include <stdint.h>
#include "utility/span.h"

class OFDM_Demod;

void RenderSourceBuffer(tcb::span<const std::complex<int16_t>> buf_raw);
void RenderOFDMDemodulator(OFDM_Demod& demod);