// This application produces a dummy OFDM signal with mode I,II,III,IV parameters
// No information is encoded in this signal
// It is only used to test if the OFDM demodulator is working correctly

#include <stdio.h>
#define _USE_MATH_DEFINES
#include <cmath>
#include <complex>
#include <vector>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif

#include "modules/ofdm/ofdm_modulator.h"
#include "modules/ofdm/ofdm_params.h"
#include "modules/ofdm/dab_prs_ref.h"
#include "modules/ofdm/dab_ofdm_params_ref.h"
#include "modules/ofdm/dab_mapper_ref.h"
#include "modules/ofdm/ofdm_dsp.h"
#include "utility/getopt/getopt.h"
#include "utility/span.h"

// scrambler that is used for DVB transmissions
class Scrambler 
{
private:
    uint16_t reg = 0;
public:
    uint16_t syncword = 0b0000000010101001;
    void Reset() {
        reg = syncword;
    }
    uint8_t Process() {
        uint8_t v = static_cast<uint8_t>(
            ((reg ^ (reg << 1)) >> 8) & 0xFF);
        reg = (reg << 8) | v;
        return v;
    }
};

void ApplyFrequencyShift(
    tcb::span<const std::complex<int16_t>> x, tcb::span<std::complex<int16_t>> y, 
    const float frequency, const float Ts=1.0f/2.048e6)
{
    // apply_pll_scalar(x, y, frequency, 0.0f, FIXED_POINT_SCALING);
    apply_pll_avx2(x, y, frequency, 0.0f, FIXED_POINT_SCALING);
}

void usage() {
    fprintf(stderr, 
        "simulate_transmitter, produces OFDM data as raw IQ values\n\n"
        "\t[-M dab transmission mode (default: 1)]\n"
        "\t[-f frequency shift in Hz (default: 0)\n"
        "\t[-P (output the binary data used as placeholder)\n"
        "\t[-h (show usage)]\n"
    );
}

int main(int argc, char** argv) 
{

#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif

    int transmission_mode = 1;
    float frequency_shift = 0;
    bool print_sample_message = false;

    int opt; 
    while ((opt = getopt_custom(argc, argv, "M:f:Ph")) != -1) {
        switch (opt) {
        case 'M':
            transmission_mode = (int)(atof(optarg));
            break;
        case 'f':
            frequency_shift = (float)(atof(optarg));
            break;
        case 'P':
            print_sample_message = true; 
            break;
        case 'h':
        default:
            usage();
            return 0;
        }
    }

    if (transmission_mode <= 0 || transmission_mode > 4) {
        fprintf(stderr, "Transmission modes: I,II,III,IV are supported not (%d)\n", transmission_mode);
        return 1;
    }

    const float max_frequency_shift = 100e3f;
    if ((frequency_shift < -max_frequency_shift) || (frequency_shift > max_frequency_shift)) {
        fprintf(stderr, "Frequency shift our of maximum range |%.2f| > %.2f\n", frequency_shift, max_frequency_shift);
        return 1;
    }

    const auto params = get_DAB_OFDM_params(transmission_mode);

    auto prs_fft_ref = std::vector<std::complex<int16_t>>(params.nb_fft);
    auto carrier_mapper = std::vector<int>(params.nb_data_carriers);
    
    get_DAB_PRS_reference(transmission_mode, prs_fft_ref);
    get_DAB_mapper_ref(carrier_mapper, params.nb_fft);

    // create our single ofdm frame
    const size_t frame_size = 
        params.nb_null_period +
        params.nb_symbol_period*params.nb_frame_symbols;
    auto frame_out_buf = std::vector<std::complex<int16_t>>(frame_size);
    auto frame_tx_buf = std::vector<std::complex<uint8_t>>(frame_size);
    
    // determine the number of bits that the ofdm frame contains
    // a single carrier contains 2 bits (there are four possible dqpsk phases)
    // the PRS (phase reference symbol) doesnt contain any information
    const size_t nb_frame_bits = (params.nb_frame_symbols-1)*params.nb_data_carriers*2;
    const size_t nb_frame_bytes = nb_frame_bits/8;
    auto frame_bytes_buf = std::vector<uint8_t>(nb_frame_bytes);

    uint16_t scrambler_code_word = 0b0000000010101001;
    auto scrambler = Scrambler();
    scrambler.Reset();
    for (int i = 0; i < nb_frame_bytes; i++) {
        frame_bytes_buf[i] = scrambler.Process();
    }

    // if we are only interested in printing the source data
    if (print_sample_message) {
        fprintf(stderr, "Outputing %zu bytes\n", frame_bytes_buf.size());
        fwrite(frame_bytes_buf.data(), sizeof(uint8_t), frame_bytes_buf.size(), stdout);
        return 0;
    }

    // perform OFDM modulation 
    auto ofdm_mod = OFDM_Modulator(params, prs_fft_ref);
    auto res = ofdm_mod.ProcessBlock(frame_out_buf, frame_bytes_buf);
    if (!res) {
        fprintf(stderr, "Failed to create the OFDM frame\n");
        return 1;
    }

    ApplyFrequencyShift(frame_out_buf, frame_out_buf, frequency_shift);
    for (int i = 0; i < frame_size; i++) {
        const int16_t I = frame_out_buf[i].real();
        const int16_t Q = frame_out_buf[i].imag();
        // NOTE: We multiply by 3/4 to prevent overflow when down casting
        const uint8_t I0 = static_cast<uint8_t>((I/5)*3 + 127);
        const uint8_t Q0 = static_cast<uint8_t>((Q/5)*3 + 127);
        frame_tx_buf[i] = std::complex<uint8_t>(I0, Q0);
    }

    while (true) {
        const size_t N = frame_tx_buf.size();
        const size_t nb_write = fwrite(frame_tx_buf.data(), sizeof(std::complex<uint8_t>), N, stdout);
        if (nb_write != N) {
            fprintf(stderr, "Failed to write out frame %zu/%zu\n", nb_write, N);
            break;
        }
    }

    return 0;
}
