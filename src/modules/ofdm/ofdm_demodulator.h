#pragma once

#include <stdint.h>
#include <complex>
#include <thread>
#include <mutex>
#include <vector>
#include <memory>

#include "ofdm_params.h"
#include "utility/reconstruction_buffer.h"
#include "utility/circular_buffer.h"
#include "utility/observable.h"
#include "utility/span.h"
#include "viterbi_config.h"

typedef struct kiss_fft_state* kiss_fft_cfg;

class OFDM_Demod_Pipeline_Thread;
class OFDM_Demod_Coordinator_Thread;

struct OFDM_Demod_Config {
    struct {
        float update_beta = 0.95f;
        int nb_samples = 100;
        int nb_decimate = 5;
    } signal_l1;
    struct {
        float thresh_null_start = 0.35f;
        float thresh_null_end = 0.75f;
    } null_l1_search;
    struct {
        // fine freq sync
        float fine_freq_update_beta = 0.9f;
        // coarse freq sync
        bool is_coarse_freq_correction = true;
        int max_coarse_freq_correction = 20000;
        float coarse_freq_slow_beta = 0.1f;
        // fine time sync
        float impulse_peak_threshold_db = 20.0f;
        float impulse_peak_distance_probability = 0.15f;
    } sync;
    bool is_update_tii_sym_mag = false;
    struct {
        bool is_update = false;
        float update_beta = 0.1f;
    } data_sym_mag;
};

class OFDM_Demod 
{
public:
    enum State {
        FINDING_NULL_POWER_DIP,
        READING_NULL_AND_PRS,
        RUNNING_COARSE_FREQ_SYNC,
        RUNNING_FINE_TIME_SYNC,
        READING_SYMBOLS,
    };
private:
    bool is_running;
    OFDM_Demod_Config cfg;
    State state;
    const OFDM_Params params;
    // statistics
    int total_frames_read;
    int total_frames_desync;
    // time and frequency correction
    std::mutex mutex_freq_fine_offset;
    float freq_coarse_offset;
    float freq_fine_offset;
    int fine_time_offset;
    // null power dip search
    bool is_null_start_found;
    bool is_null_end_found;
    float signal_l1_average;
    // fft
    kiss_fft_cfg fft_cfg;
    kiss_fft_cfg ifft_cfg;
    // threads
    std::unique_ptr<OFDM_Demod_Coordinator_Thread>           coordinator_thread;
    std::vector<std::unique_ptr<OFDM_Demod_Pipeline_Thread>> pipelines;
    std::vector<std::unique_ptr<std::thread>>                threads;
    // callback for when ofdm is completed
    Observable<tcb::span<const viterbi_bit_t>> obs_on_ofdm_frame;
    // Joint memory allocation block
    std::unique_ptr<uint8_t[]> joint_data_block;
    // 1. pipeline reader double buffer
    ReconstructionBuffer<std::complex<int16_t>> active_buffer;
    ReconstructionBuffer<std::complex<int16_t>> inactive_buffer;
    tcb::span<std::complex<int16_t>> active_buffer_data;
    tcb::span<std::complex<int16_t>> inactive_buffer_data;
    // 2. fine time and coarse frequency synchronisation using time/frequency correlation
    CircularBuffer<std::complex<int16_t>> null_power_dip_buffer;
    ReconstructionBuffer<std::complex<int16_t>> correlation_time_buffer;
    tcb::span<std::complex<int16_t>>    null_power_dip_buffer_data;
    tcb::span<std::complex<int16_t>>    correlation_time_buffer_data;
    tcb::span<float>                  correlation_impulse_response;
    tcb::span<float>                  correlation_frequency_response;
    tcb::span<std::complex<int16_t>>    correlation_fft_buffer;
    tcb::span<std::complex<int16_t>>    correlation_prs_fft_reference;
    tcb::span<std::complex<int16_t>>    correlation_prs_time_reference;
    // 3. pipeline demodulation
    tcb::span<std::complex<int16_t>>    pipeline_fft_buffer;
    tcb::span<std::complex<int16_t>>    pipeline_dqpsk_vec_buffer;
    tcb::span<viterbi_bit_t>          pipeline_out_bits;
    // 4. carrier frequency deinterleaving
    tcb::span<int> carrier_mapper;
public:
    OFDM_Demod(
        const OFDM_Params _params, 
        tcb::span<const std::complex<int16_t>> _prs_fft_ref, 
        tcb::span<const int> _carrier_mapper,
        int nb_desired_threads=0);
    ~OFDM_Demod();
    // threads use lambdas which take in the this pointer
    // therefore we disable move/copy semantics to preservce its memory location
    OFDM_Demod(OFDM_Demod&) = delete;
    OFDM_Demod(OFDM_Demod&&) = delete;
    OFDM_Demod& operator=(OFDM_Demod&) = delete;
    OFDM_Demod& operator=(OFDM_Demod&&) = delete;
    void Process(tcb::span<const std::complex<int16_t>> block);
    void Reset();
public:
    OFDM_Params GetOFDMParams() const { return params; }
    State GetState() const { return state; }
    auto& GetConfig() { return cfg; }
    float GetSignalAverage() const { return signal_l1_average; }
    float GetFineFrequencyOffset() const { return freq_fine_offset; }
    float GetCoarseFrequencyOffset() const { return freq_coarse_offset; }
    float GetNetFrequencyOffset() const { return freq_fine_offset + freq_coarse_offset; }
    int GetFineTimeOffset() const { return fine_time_offset; }
    int GetTotalFramesRead() const { return total_frames_read; }
    int GetTotalFramesDesync() const { return total_frames_desync; }
    auto GetFrameDataVec() { return pipeline_dqpsk_vec_buffer; }
    auto GetFrameDataBits() { return pipeline_out_bits; }
    tcb::span<float> GetImpulseResponse() { return correlation_impulse_response; }
    tcb::span<float> GetCoarseFrequencyResponse() { return correlation_frequency_response; }
    tcb::span<const std::complex<int16_t>> GetCorrelationTimeBuffer() { return correlation_time_buffer; }
    auto& On_OFDM_Frame() { return obs_on_ofdm_frame; }
private:
    size_t FindNullPowerDip(tcb::span<const std::complex<int16_t>> buf);
    size_t ReadNullPRS(tcb::span<const std::complex<int16_t>> buf);
    size_t RunCoarseFreqSync(tcb::span<const std::complex<int16_t>> buf);
    size_t RunFineTimeSync(tcb::span<const std::complex<int16_t>> buf);
    size_t ReadSymbols(tcb::span<const std::complex<int16_t>> buf);
private:
    void CoordinatorThread();
    void PipelineThread(OFDM_Demod_Pipeline_Thread& thread_data, OFDM_Demod_Pipeline_Thread* dependent_thread_data);
private:
    float ApplyPLL(
        tcb::span<const std::complex<int16_t>> x, tcb::span<std::complex<int16_t>> y, 
        const float freq_offset, const float dt0=0);
    float CalculateTimeOffset(const size_t i, const float freq_offset);
    float CalculateCyclicPhaseError(tcb::span<const std::complex<int16_t>> sym);
    float CalculateFineFrequencyError(const float cyclic_phase_error);
    void CalculateDQPSK(
        tcb::span<const std::complex<int16_t>> in0, tcb::span<const std::complex<int16_t>> in1, 
        tcb::span<std::complex<int16_t>> out_vec);
    void CalculateViterbiBits(tcb::span<const std::complex<int16_t>> vec_buf, tcb::span<viterbi_bit_t> bit_buf);
    void CalculateFFT(tcb::span<const std::complex<int16_t>> fft_in, tcb::span<std::complex<int16_t>> fft_out);
    void CalculateIFFT(tcb::span<const std::complex<int16_t>> fft_in, tcb::span<std::complex<int16_t>> fft_out);
    void CalculateRelativePhase(tcb::span<const std::complex<int16_t>> fft_in, tcb::span<std::complex<int16_t>> arg_out);
    void CalculateMagnitude(tcb::span<const std::complex<int16_t>> fft_buf, tcb::span<float> mag_buf);
    float CalculateL1Average(tcb::span<const std::complex<int16_t>> block);
    void UpdateSignalAverage(tcb::span<const std::complex<int16_t>> block);
    void UpdateFineFrequencyOffset(const float delta);
};

