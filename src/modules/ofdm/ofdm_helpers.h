#pragma once

#include <stdint.h>
#include <complex>
#include <memory>
#include "ofdm_demodulator.h"
#include "dab_ofdm_params_ref.h"
#include "dab_prs_ref.h"
#include "dab_mapper_ref.h"
#include "utility/span.h"

static void ConvertRawToExpected(tcb::span<const std::complex<uint8_t>> x, tcb::span<std::complex<int16_t>> y) {
	const int N = (int)x.size();
	for (int i = 0; i < N; i++) {
		auto& v = x[i];
		const int16_t I = (int16_t)(v.real()) - 127;
		const int16_t Q = (int16_t)(v.imag()) - 127;
		y[i] = { I, Q };
	}
}

static void ConvertFloatToFixed(tcb::span<const std::complex<float>> x, tcb::span<std::complex<int16_t>> y) {
	const float FIXED_POINT_SCALING = 128.0f;
	const auto N = x.size();
	for (int i = 0; i < N; i++) {
		y[i] = {
			(int16_t)(x[i].real() * FIXED_POINT_SCALING),
			(int16_t)(x[i].imag() * FIXED_POINT_SCALING)
		};
	}
}

static std::unique_ptr<OFDM_Demod> Create_OFDM_Demodulator(const int transmission_mode, const int total_threads=0) {
	const OFDM_Params ofdm_params = get_DAB_OFDM_params(transmission_mode);
	auto ofdm_prs_ref = std::vector<std::complex<int16_t>>(ofdm_params.nb_fft);
	get_DAB_PRS_reference(transmission_mode, ofdm_prs_ref);
	auto ofdm_mapper_ref = std::vector<int>(ofdm_params.nb_data_carriers);
	get_DAB_mapper_ref(ofdm_mapper_ref, ofdm_params.nb_fft);
	auto ofdm_demod = std::make_unique<OFDM_Demod>(ofdm_params, ofdm_prs_ref, ofdm_mapper_ref, total_threads);
	return std::move(ofdm_demod);
}
