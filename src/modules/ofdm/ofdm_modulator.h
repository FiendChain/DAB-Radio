#include <stdint.h>
#include <complex>
#include <vector>
#include "ofdm_params.h"
#include "utility/span.h"

typedef struct kiss_fft_state* kiss_fft_cfg;

// simulate a OFDM transmitter using one of the DAB transmission modes
// this will have a sampling rate of 2.048MHz
class OFDM_Modulator 
{
private:
    kiss_fft_cfg ifft_cfg;
    const OFDM_Params params;

    const size_t frame_out_size;
    const size_t data_in_size;

    std::vector<std::complex<int16_t>> prs_fft_ref;
    std::vector<std::complex<int16_t>> prs_time_ref;

    // frequency domain buffers used for dqpsk encoding
    std::vector<std::complex<int16_t>> last_sym_fft;
    std::vector<std::complex<int16_t>> curr_sym_fft;
public:
    OFDM_Modulator(
        const OFDM_Params _params, 
        tcb::span<const std::complex<int16_t>> _prs_fft_ref);
    ~OFDM_Modulator();
    bool ProcessBlock(
        tcb::span<std::complex<int16_t>> frame_out_buf, 
        tcb::span<const uint8_t> data_in_buf);
private:
    void CreateDataSymbol(
        tcb::span<const uint8_t> sym_data_in, 
        tcb::span<std::complex<int16_t>> sym_out);
};