#include "ofdm_modulator.h"
#include <kiss_fft.h>
#include <algorithm>
#include <stdio.h>

OFDM_Modulator::OFDM_Modulator(
    const OFDM_Params _params, 
    tcb::span<const std::complex<int16_t>> _prs_fft_ref)
:   params(_params),
    frame_out_size(_params.nb_null_period + _params.nb_symbol_period*_params.nb_frame_symbols),
    data_in_size((_params.nb_frame_symbols-1)*_params.nb_data_carriers*2/8)
{
    ifft_cfg = kiss_fft_alloc((int)params.nb_fft, true, NULL, NULL);

    // interleave the bits for a OFDM symbol containing N data carriers
    prs_fft_ref.resize(params.nb_fft);
    prs_time_ref.resize(params.nb_symbol_period);

    const int16_t PRESCALE = 32;
    // NOTE: Scale PRS_FFT so it doesn't get zeroed by IFFT 
    for (int i = 0; i < params.nb_fft; i++) {
        prs_fft_ref[i] = _prs_fft_ref[i] * PRESCALE;
    }

    // create our time domain prs symbol with the cyclic prefix
    {
        auto* buf = &prs_time_ref[params.nb_cyclic_prefix];
        kiss_fft(ifft_cfg, (kiss_fft_cpx*)prs_fft_ref.data(), (kiss_fft_cpx*)buf);
        for (int i = 0; i < params.nb_cyclic_prefix; i++) {
            prs_time_ref[i] = prs_time_ref[params.nb_fft+i];
        }
    }

    // NOTE: Downscale PRS_FFT back to original
    for (int i = 0; i < params.nb_fft; i++) {
        prs_fft_ref[i] = prs_fft_ref[i] / PRESCALE;
    }

    for (int i = 0; i < params.nb_fft; i++) {
        if (prs_fft_ref[i] != _prs_fft_ref[i]) {
            fprintf(stderr, "Ah fuck it didn't rescale properly\n");
            break;
        }
    }

    last_sym_fft.resize(params.nb_fft);
    curr_sym_fft.resize(params.nb_fft);

    for (int i = 0; i < params.nb_fft; i++) {
        curr_sym_fft[i] = std::complex<int16_t>(0,0);
        last_sym_fft[i] = std::complex<int16_t>(0,0);
    }
}

OFDM_Modulator::~OFDM_Modulator() 
{
    kiss_fft_free(ifft_cfg);
}

bool OFDM_Modulator::ProcessBlock(
    tcb::span<std::complex<int16_t>> frame_out_buf, 
    tcb::span<const uint8_t> data_in_buf)
{
    const size_t nb_frame_out = frame_out_buf.size();
    const size_t nb_data_in = data_in_buf.size();
    // invalid buffer sizes
    if (nb_data_in != data_in_size) {
        return false;
    }
    if (nb_frame_out != frame_out_size) {
        return false;
    }

    // null period
    for (int i = 0; i < params.nb_null_period; i++) {
        frame_out_buf[i] = std::complex<int16_t>(0,0);
    }

    // prs symbol
    for (int i = 0; i < params.nb_symbol_period; i++) {
        frame_out_buf[i+params.nb_null_period] = prs_time_ref[i];
    }

    // seed the dqpsk fft buffers
    for (int i = 0; i < params.nb_fft; i++) {
        last_sym_fft[i] = prs_fft_ref[i];
    }

    // data symbols
    {
        const size_t nb_data_symbols = params.nb_frame_symbols-1;
        const size_t nb_data_in = params.nb_data_carriers*2/8; 
        const size_t nb_out = params.nb_symbol_period;
        // account for the PRS (phase reference symbol)
        auto* data_symbols = &frame_out_buf[params.nb_null_period + params.nb_symbol_period];

        for (int i = 0; i < nb_data_symbols; i++) {
            const auto* sym_data_in = &data_in_buf[i*nb_data_in];
            auto* sym_out = &data_symbols[i*params.nb_symbol_period];
            CreateDataSymbol(
                {sym_data_in, nb_data_in}, 
                {sym_out, params.nb_symbol_period});
        }
    }

    return true;
}

void OFDM_Modulator::CreateDataSymbol(
    tcb::span<const uint8_t> sym_data_in, 
    tcb::span<std::complex<int16_t>> sym_out)
{
    const size_t nb_data_in = sym_data_in.size(); 
    const size_t nb_out = sym_out.size();

    constexpr int16_t A = 90;
    constexpr int16_t A_MAG = 127;
    static std::complex<int16_t> PHASE_MAP[4] = {
        {-A,-A}, {A,-A}, {A,A}, {-A,A}};

    // Create raw fft bins
    {
        // create fft for -F/2 <= f < 0
        size_t curr_carrier = params.nb_fft - params.nb_data_carriers/2;
        for (int i = 0; i < nb_data_in/2; i++) {
            const uint8_t b = sym_data_in[i];
            curr_sym_fft[curr_carrier++] = PHASE_MAP[((b >> 0) & 0b11)];
            curr_sym_fft[curr_carrier++] = PHASE_MAP[((b >> 2) & 0b11)];
            curr_sym_fft[curr_carrier++] = PHASE_MAP[((b >> 4) & 0b11)];
            curr_sym_fft[curr_carrier++] = PHASE_MAP[((b >> 6) & 0b11)];
        }

        // create fft for 0 < f <= F/2 
        curr_carrier = 1;
        for (int i = 0; i < nb_data_in/2; i++) {
            const size_t j = nb_data_in/2 + i;
            const uint8_t b = sym_data_in[j];
            curr_sym_fft[curr_carrier++] = PHASE_MAP[((b >> 0) & 0b11)];
            curr_sym_fft[curr_carrier++] = PHASE_MAP[((b >> 2) & 0b11)];
            curr_sym_fft[curr_carrier++] = PHASE_MAP[((b >> 4) & 0b11)];
            curr_sym_fft[curr_carrier++] = PHASE_MAP[((b >> 6) & 0b11)];
        }
    }


    // get the dqpsk
    // arg(z0*z1) = arg(z0) + arg(z1)
    {
        for (int i = 0; i < params.nb_data_carriers/2; i++) {
            const size_t j = params.nb_fft - params.nb_data_carriers/2 + i;
            curr_sym_fft[j] = last_sym_fft[j] * curr_sym_fft[j];
            curr_sym_fft[j] /= A_MAG;
        }

        for (int i = 0; i < params.nb_data_carriers/2; i++) {
            const size_t j = 1+i;
            curr_sym_fft[j] = last_sym_fft[j] * curr_sym_fft[j];
            curr_sym_fft[j] /= A_MAG;
        }
    }

    const int16_t PRESCALE = 32;
    // NOTE: we pass in the data as A*PRESCALE so it doesn't get zeroed by IFFT
    for (int i = 0; i < params.nb_fft; i++) {
        curr_sym_fft[i] *= PRESCALE;
    }

    // get ifft of symbol
    {
        auto* buf = &sym_out[params.nb_cyclic_prefix];
        kiss_fft(ifft_cfg, (kiss_fft_cpx*)curr_sym_fft.data(), (kiss_fft_cpx*)buf);
    }

    // NOTE: rescale curr_sym_fft back down to +-A
    for (int i = 0; i < params.nb_fft; i++) {
        curr_sym_fft[i] /= PRESCALE;
    }

    // create cyclic prefix
    for (int i = 0; i < params.nb_cyclic_prefix; i++) {
        sym_out[i] = sym_out[i+params.nb_fft];
    }

    // swap fft buffers
    std::swap(last_sym_fft, curr_sym_fft);
}