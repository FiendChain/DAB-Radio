#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>

#include "ofdm_demodulator.h"
#include "ofdm_demodulator_threads.h"
#include <kiss_fft.h>
#include "ofdm_dsp.h"

#include "utility/joint_allocate.h"

#define PROFILE_ENABLE 1
#include "utility/profiler.h"

static constexpr float Fs = 2.048e6;
static constexpr float Ts = 1.0f/Fs;

#define USE_PLL_TABLE 0
#if USE_PLL_TABLE
#include "quantized_oscillator.h"
// NOTE: For any decent phase locked loop performance the frequency resolution <= 5Hz
//       This puts a minimum bound on the table size as 2048000*8/5 = 3.27MB
//       This ends up being absolutely horrible due to cache misses
//       Verified by Intel V Profiler memory access analysis - high cache miss count
// Frequency resolution of table can't be too small otherwise we end up with a large table
// This could cause inefficient memory access and cache misses
static auto PLL_TABLE = QuantizedOscillator(5, (int)(Fs));
#endif

// DOC: docs/DAB_implementation_in_SDR_detailed.pdf
// All of the implmentation details are based on the specified document
OFDM_Demod::OFDM_Demod(
    const OFDM_Params _params, 
    tcb::span<const std::complex<int16_t>> _prs_fft_ref, 
    tcb::span<const int> _carrier_mapper,
    int nb_desired_threads)
:   params(_params), 
    null_power_dip_buffer(null_power_dip_buffer_data),
    correlation_time_buffer(correlation_time_buffer_data),
    active_buffer(active_buffer_data),
    inactive_buffer(inactive_buffer_data)
{
    joint_data_block = AllocateJoint(
        carrier_mapper,                 params.nb_data_carriers, 
        // Fine time correlation and coarse frequency correction
        null_power_dip_buffer_data,     params.nb_null_period,
        correlation_time_buffer_data,   params.nb_null_period + params.nb_symbol_period,
        correlation_prs_fft_reference,  params.nb_fft,
        correlation_prs_time_reference, params.nb_fft,
        correlation_impulse_response,   params.nb_fft,
        correlation_frequency_response, params.nb_fft,
        correlation_fft_buffer,         params.nb_fft, 
        // Double buffer ingest so our reader thread isn't blocked and drops samples from rtl_sdr.exe
        active_buffer_data,             params.nb_frame_symbols*params.nb_symbol_period + params.nb_null_period,
        inactive_buffer_data,           params.nb_frame_symbols*params.nb_symbol_period + params.nb_null_period,
        // Data structures to read all 76 symbols + NULL symbol and perform demodulation 
        pipeline_fft_buffer,            (params.nb_frame_symbols+1)*params.nb_fft,
        pipeline_dqpsk_vec_buffer,      (params.nb_frame_symbols-1)*params.nb_fft,
        pipeline_out_bits,              (params.nb_frame_symbols-1)*params.nb_data_carriers*2
    );

    // NOTE: Using the FFT as an IFFT will cause the result to be reversed in time
    // Separate configurations for FFT and IFFT
    fft_cfg = kiss_fft_alloc((int)params.nb_fft, false, 0, 0);
    ifft_cfg = kiss_fft_alloc((int)params.nb_fft, true, 0, 0);

    // Initial state of demodulator
    state = State::FINDING_NULL_POWER_DIP;
    total_frames_desync = 0;
    total_frames_read = 0;
    freq_coarse_offset = 0;
    freq_fine_offset = 0;
    fine_time_offset = 0;
    is_null_start_found = false;
    is_null_end_found = false;
    signal_l1_average = 0;
    is_running = true;

    // Fine time synchronisation
    // Correlation in time domain is the conjugate product in frequency domain
    for (int i = 0; i < params.nb_fft; i++) {
        correlation_prs_fft_reference[i] = std::conj(_prs_fft_ref[i]);
    }

    // Coarse frequency synchronisation
    // Correlation in frequency domain is the conjugate product in time domain
    CalculateRelativePhase(_prs_fft_ref, correlation_prs_time_reference);
    CalculateIFFT(correlation_prs_time_reference, correlation_prs_time_reference);
    for (int i = 0; i < params.nb_fft; i++) {
        correlation_prs_time_reference[i] = std::conj(correlation_prs_time_reference[i]);
    }

    // DOC: ETSI EN 300 401
    // Referring to clause 14.5 - QPSK symbol mapper
    // Our subcarriers for each symbol are distributed according to a one to one rule
    std::copy_n(_carrier_mapper.begin(), params.nb_data_carriers, carrier_mapper.begin());

    // Setup our multithreaded processing pipeline
    coordinator_thread = std::make_unique<OFDM_Demod_Coordinator_Thread>();
    {
        const int nb_syms = (int)params.nb_frame_symbols+1;
        const int total_system_threads = (int)std::thread::hardware_concurrency();
        int nb_threads = 0; 
        // Manually set number of threads
        if (nb_desired_threads > 0) {
            nb_threads = std::min(nb_syms, nb_desired_threads);
        // Automatically determine
        } else {
            nb_threads = std::min(nb_syms, total_system_threads);
            // NOTE: If we have alot of physical cores, then reducing the number of pipeline threads
            //       actually improves performance slightly since we reduce thread contention with other threads
            //       Thread contention causes some pipeline threads to start late or take longer
            //       Since the coordinator thread has to wait for the slowest pipeline thread
            //       minimising inter-thread variance by reducing the number of assigned threads actually improves performance
            if (nb_threads > 8) {
                nb_threads -= 1;
            } 
        }

        int symbol_start = 0;    
        for (int i = 0; i < nb_threads; i++) {
            const bool is_last_thread = (i == (nb_threads-1));

            const int nb_syms_remain = (nb_syms-symbol_start);
            const int nb_threads_remain = (nb_threads-i);
            const int nb_syms_in_thread = (int)std::ceil((float)nb_syms_remain / (float)nb_threads_remain);
            const int symbol_end = is_last_thread ? nb_syms : (symbol_start+nb_syms_in_thread);
            pipelines.emplace_back(std::move(std::make_unique<OFDM_Demod_Pipeline_Thread>(
                symbol_start, symbol_end
            )));
            symbol_start = symbol_end;
        }
    }
    threads.emplace_back(std::move(std::make_unique<std::thread>(
        [this]() {
            PROFILE_TAG_THREAD("OFDM_Demod::CoordinatorThread");
            while (is_running) {
                CoordinatorThread();
            }
            coordinator_thread->SignalEnd();
        }
    )));

    for (int i = 0; i < pipelines.size(); i++) {
        auto& pipeline = *(pipelines[i].get());
        auto* dependent_pipeline = ((i+1) >= pipelines.size()) ? NULL : pipelines[i+1].get();
        threads.emplace_back(std::move(std::make_unique<std::thread>(
            [this, &pipeline, dependent_pipeline]() {
                PROFILE_TAG_THREAD("OFDM_Demod::PipelineThread");

                // Give custom data to profiler
                union Data { 
                    struct { uint32_t start, end; } fields; 
                    uint64_t data;
                } X;
                X.fields.start = (int)pipeline.GetSymbolStart();
                X.fields.end   = (int)pipeline.GetSymbolEnd();
                PROFILE_TAG_DATA_THREAD(X.data);
                while (is_running) {
                    PipelineThread(pipeline, dependent_pipeline);
                }
            }
        )));
    }
}

OFDM_Demod::~OFDM_Demod() {
    // join all threads 
    coordinator_thread->Wait();
    is_running = false;
    coordinator_thread->Stop();
    for (auto& pipeline: pipelines) {
        pipeline->Stop();
    }
    for (auto& thread: threads) {
        thread->join();
    }

    // fft/ifft buffers
    kiss_fft_free(fft_cfg);
    kiss_fft_free(ifft_cfg);
}

// Thread 1: Read frame data at start of frame
// Clause 3.12.1: Symbol timing synchronisation
// Clause 3.13.2 Integral frequency offset estimation
// Clause 3.12.2: Frame synchronisation
void OFDM_Demod::Process(tcb::span<const std::complex<int16_t>> buf) {
    PROFILE_TAG_THREAD("OFDM_Demod::ProcessThread");
    PROFILE_ENABLE_TRACE_LOGGING(true);
    PROFILE_ENABLE_TRACE_LOGGING_CONTINUOUS(true);
    PROFILE_BEGIN_FUNC();

    UpdateSignalAverage(buf);

    const size_t N = buf.size();
    size_t curr_index = 0;
    while (curr_index < N) {
        auto* block = &buf[curr_index];
        const size_t N_remain = N-curr_index;

        switch (state) {

        // Clause 3.12.1: Symbol timing synchronisation
        case State::FINDING_NULL_POWER_DIP:
            curr_index += FindNullPowerDip({block, N_remain});
            break;
        
        case State::READING_NULL_AND_PRS:
            curr_index += ReadNullPRS({block, N_remain});
            break;
        
        // Clause 3.13.2 Integral frequency offset estimation
        case State::RUNNING_COARSE_FREQ_SYNC:
            curr_index += RunCoarseFreqSync({block, N_remain});
            break;
        
        // Clause 3.12.2: Frame synchronisation
        case State::RUNNING_FINE_TIME_SYNC:
            curr_index += RunFineTimeSync({block, N_remain});
            break;
        
        case State::READING_SYMBOLS:
            curr_index += ReadSymbols({block, N_remain});
            break;
        }
    }
}

void OFDM_Demod::Reset() {
    PROFILE_BEGIN_FUNC();
    state = State::FINDING_NULL_POWER_DIP;
    correlation_time_buffer.SetLength(0);
    total_frames_desync++;

    // NOTE: We also reset fine frequency synchronisation since an incorrect value
    // can reduce performance of fine time synchronisation using the impulse response
    freq_coarse_offset = 0;
    freq_fine_offset = 0;
    fine_time_offset = 0;
    signal_l1_average = 0;
}

size_t OFDM_Demod::FindNullPowerDip(tcb::span<const std::complex<int16_t>> buf) {
    PROFILE_BEGIN_FUNC();
    // Clause 3.12.2 - Frame synchronisation using power detection
    // we run this if we dont have an initial estimate for the prs index
    // This can occur if:
    //      1. We just started the demodulator and need a quick estimate of OFDM start
    //      2. The PRS impulse response didn't have a sufficiently large peak

    // We analyse the average power of the signal using blocks of size K
    const int N = (int)buf.size();
    const int K = (int)cfg.signal_l1.nb_samples;
    const int M = N-K;

    const float null_start_thresh = signal_l1_average * cfg.null_l1_search.thresh_null_start;
    const float null_end_thresh = signal_l1_average * cfg.null_l1_search.thresh_null_end;

    // if the loop doesn't exit then we copy all samples into circular buffer
    int nb_read = N;
    for (int i = 0; i < M; i+=K) {
        auto* block = &buf[i];
        const float l1_avg = CalculateL1Average({block, (size_t)K});
        if (is_null_start_found) {
            if (l1_avg > null_end_thresh) {
                is_null_end_found = true;
                nb_read = i+K;
                break;
            }
        } else {
            if (l1_avg < null_start_thresh) {
                is_null_start_found = true;
            }
        }
    }

    null_power_dip_buffer.ConsumeBuffer({buf.data(), (size_t)nb_read}, true);
    if (!is_null_end_found) {
        return (size_t)nb_read;
    }

    // Copy null symbol into correlation buffer
    // This is done since our captured null symbol may actually contain parts of the PRS 
    // We do this so we can guarantee the full PRS is attained after fine time synchronisation using correlation
    const size_t L = null_power_dip_buffer.Length();
    const size_t start_index = null_power_dip_buffer.GetIndex();
    for (int i = 0; i < L; i++) {
        const size_t j = i+start_index;
        correlation_time_buffer[i] = null_power_dip_buffer[j];
    }
    
    is_null_start_found = false;
    is_null_end_found = false;
    correlation_time_buffer.SetLength(L);
    null_power_dip_buffer.SetLength(0);
    state = State::READING_NULL_AND_PRS;

    return (size_t)nb_read;
}

size_t OFDM_Demod::ReadNullPRS(tcb::span<const std::complex<int16_t>> buf) {
    PROFILE_BEGIN_FUNC();
    const size_t nb_read = correlation_time_buffer.ConsumeBuffer(buf);
    if (!correlation_time_buffer.IsFull()) {
        return nb_read;
    }

    state = State::RUNNING_COARSE_FREQ_SYNC;
    return nb_read;
}

size_t OFDM_Demod::RunCoarseFreqSync(tcb::span<const std::complex<int16_t>> buf) {
    PROFILE_BEGIN_FUNC();
    // Clause: 3.13.2 Integral frequency offset estimation
    if (!cfg.sync.is_coarse_freq_correction) {
        freq_coarse_offset = 0;
        state = State::RUNNING_FINE_TIME_SYNC;
        return 0;
    }

    auto corr_time_buf = tcb::span(correlation_time_buffer);
    auto prs_sym = corr_time_buf.subspan(params.nb_null_period, params.nb_symbol_period);

    // To find the coarse frequency error correlate the FFT of the received and reference PRS
    // To mitigate effect of phase shifts we instead correlate the complex difference between consecutive FFT bins
    // arg(~z0*z1) = arg(z1)-arg(z0)

    // Step 1: Get FFT of received PRS 
    std::copy_n(prs_sym.begin(), prs_sym.size(), correlation_fft_buffer.begin());

    // NOTE: We scale the data up otherwise it wont produce an FFT
    for (int i = 0; i < params.nb_fft; i++) {
        correlation_fft_buffer[i] *= FIXED_POINT_SCALING/4;
    }
    CalculateFFT(correlation_fft_buffer, correlation_fft_buffer);

    // Step 2: Get complex difference between consecutive bins
    CalculateRelativePhase(correlation_fft_buffer, correlation_fft_buffer);

    // Step 3: Get IFFT so we can do correlation in frequency domain via product in time domain
    CalculateIFFT(correlation_fft_buffer, correlation_fft_buffer);

    // Step 4: Conjugate product in time domain
    //         NOTE: correlation_prs_time_reference is already the conjugate
    for (int i = 0; i < params.nb_fft; i++) {
        correlation_fft_buffer[i] *= correlation_prs_time_reference[i];
    }

    // Step 5: Get FFT to get correlation in frequency domain
    CalculateFFT(correlation_fft_buffer, correlation_fft_buffer);

    // Step 6: Get magnitude spectrum so we can find the correlation peak
    CalculateMagnitude(correlation_fft_buffer, correlation_frequency_response);

    // Step 7: Find the peak in our maximum coarse frequency error window
    // NOTE: A zero frequency error corresponds to a peak at nb_fft/2
    const int max_carrier_offset = cfg.sync.max_coarse_freq_correction / (int)params.freq_carrier_spacing;
    const int M = (int)params.nb_fft/2;
    int max_index = -max_carrier_offset;
    float max_value = correlation_frequency_response[max_index+M];
    for (int i = -max_carrier_offset; i <= max_carrier_offset; i++) {
        const int fft_index = i+M;
        const float value = correlation_frequency_response[fft_index];
        if (value > max_value) {
            max_value = value;
            max_index = i;
        }
    }

    // Step 8: Determine the coarse frequency offset 
    // NOTE: We get the frequency offset in terms of FFT bins which we convert to Hz
    const float ofdm_freq_spacing = (float)(params.freq_carrier_spacing);
    const float predicted_freq_coarse_offset = -(float)max_index * ofdm_freq_spacing;
    const float error = predicted_freq_coarse_offset-freq_coarse_offset;

    // Step 9: Determine if this is an large or small correction
    // Case A: If we have a large correction, we need to immediately update or subsequent processing
    //         will be performed on a horribly out of sync signal
    // Case B: If we have a small correction, i.e. within one FFT bin, then slowly update
    //         This is because we may end up in a state where the offset lies between two adjacent FFT bins
    //         This can cause the coarse frequency correction to oscillate between those two adjacent bins
    //         We can reduce the update rate so the coarse frequency correction doesn't fluctuate too much
    const bool is_large_correction = std::abs(error) > ofdm_freq_spacing*1.5f;
    const float beta = is_large_correction ? 1.0f : cfg.sync.coarse_freq_slow_beta;
    const float delta = beta*error;

    // Step 10: Update the coarse frequency offset
    freq_coarse_offset += delta;

    // Step 11: Counter adjust the fine frequency offset
    // In a near locked state the coarse frequency offset may fluctuate alot if it lies between two FFT bins
    // By counter adjusting the fine frequency offset, the combined coarse and fine frequency offset will be stable
    UpdateFineFrequencyOffset(-delta);

    state = State::RUNNING_FINE_TIME_SYNC;
    return 0;
}

size_t OFDM_Demod::RunFineTimeSync(tcb::span<const std::complex<int16_t>> buf) {
    PROFILE_BEGIN_FUNC();
    // Clause 3.12.1 - Symbol timing synchronisation
    auto corr_time_buf = tcb::span(correlation_time_buffer);
    auto corr_prs_buf = corr_time_buf.subspan(params.nb_null_period, params.nb_symbol_period);

    // To synchronise to start of the PRS we calculate the impulse response 
    const float freq_offset = freq_coarse_offset + freq_fine_offset;
    std::copy_n(corr_prs_buf.begin(), params.nb_fft, correlation_fft_buffer.begin());
    // Correlation in time domain is done by doing multiplication in frequency domain
    ApplyPLL(correlation_fft_buffer, correlation_fft_buffer, freq_offset);
    CalculateFFT(correlation_fft_buffer, correlation_fft_buffer);
    for (int i = 0; i < params.nb_fft; i++) {
        correlation_fft_buffer[i] *= correlation_prs_fft_reference[i];
    }

    // Get IFFT to get our correlation result
    CalculateIFFT(correlation_fft_buffer, correlation_fft_buffer);
    for (int i = 0; i < params.nb_fft; i++) {
        const auto& v = correlation_fft_buffer[i];
        const float A = 20.0f*std::log10((float)std::abs(v));
        correlation_impulse_response[i] = A;
    }

    // calculate if we have a valid impulse response
    // if the peak is at least X dB above the mean, then we use that as our PRS starting index
    float impulse_avg = 0.0f;
    float impulse_max_value = correlation_impulse_response[0];
    int impulse_max_index = 0;
    for (int i = 0; i < params.nb_fft; i++) {
        const float peak_value = correlation_impulse_response[i];

        // We expect that the correlation peak will at least be somewhere near where we expect it
        // When we are still locking on, the impulse response may have many peaks
        // This causes spurious desyncs when one of these other peaks are very far away
        // Thus we weigh the value of the peak with its distance from the expected location
        const int expected_peak_x = (int)params.nb_cyclic_prefix;
        const int distance_from_expectation = std::abs(expected_peak_x-i);
        const float norm_distance = (float)distance_from_expectation / (float)params.nb_symbol_period;
        const float decay_weight = 1.0f - cfg.sync.impulse_peak_distance_probability;
        const float probability = 1.0f - decay_weight * norm_distance;
        const float weighted_peak_value = probability*peak_value;

        impulse_avg += peak_value;
        if (weighted_peak_value > impulse_max_value) {
            impulse_max_value = weighted_peak_value;
            impulse_max_index = i;
        }
    }
    impulse_avg /= (float)params.nb_fft;

    // If the main lobe is insufficiently powerful we do not have a valid impulse response
    // This probably means we had a severe desync and should restart
    if ((impulse_max_value - impulse_avg) < cfg.sync.impulse_peak_threshold_db) {
        Reset();
        return 0;
    }

    // The PRS correlation lobe occurs just after the cyclic prefix
    // We actually want the index at the start of the cyclic prefix, so we adjust offset for that
    const int offset = impulse_max_index - (int)params.nb_cyclic_prefix;
    const int prs_start_index = (int)params.nb_null_period + offset;
    const int prs_length = (int)params.nb_symbol_period - offset;

    inactive_buffer.SetLength((size_t)prs_length);
    for (int i = 0; i < prs_length; i++) {
        const int j = prs_start_index+i;
        inactive_buffer[i] = correlation_time_buffer[j];
    }

    correlation_time_buffer.SetLength(0);
    fine_time_offset = offset;
    state = State::READING_SYMBOLS;
    return 0;
}

size_t OFDM_Demod::ReadSymbols(tcb::span<const std::complex<int16_t>> buf) {
    PROFILE_BEGIN_FUNC();
    const size_t nb_read = inactive_buffer.ConsumeBuffer(buf);
    if (!inactive_buffer.IsFull()) {
        return nb_read;
    }

    // Copy the null symbol so we can use it in the PRS correlation step
    auto null_sym = tcb::span(inactive_buffer).last(params.nb_null_period);
    correlation_time_buffer.SetLength(params.nb_null_period);
    for (int i = 0; i < params.nb_null_period; i++) {
        correlation_time_buffer[i] = null_sym[i];
    }

    PROFILE_BEGIN(coordinator_wait);
    coordinator_thread->Wait();
    PROFILE_END(coordinator_wait);
    // double buffer
    std::swap(inactive_buffer_data, active_buffer_data);
    inactive_buffer.SetLength(0);
    // launch all our worker threads
    PROFILE_BEGIN(coordinator_start);
    coordinator_thread->Start();
    PROFILE_END(coordinator_start);

    state = State::READING_NULL_AND_PRS;
    return nb_read;
}

// Thread 2: Coordinate pipeline threads and combine fine time synchronisation results
// Clause 3.13: Frequency offset estimation and correction
// Clause 3.13.1: Fractional frequency offset estimation
void OFDM_Demod::CoordinatorThread() {
    PROFILE_BEGIN_FUNC();

    PROFILE_BEGIN(coordinator_wait_start);
    coordinator_thread->WaitStart();
    PROFILE_END(coordinator_wait_start);

    if (coordinator_thread->IsStopped()) {
        return;
    }

    PROFILE_BEGIN(pipeline_workers);

    PROFILE_BEGIN(pipeline_start);
    for (auto& pipeline: pipelines) {
        pipeline->Start();
    }
    PROFILE_END(pipeline_start);

    PROFILE_BEGIN(pipeline_wait_phase_error);
    for (auto& pipeline: pipelines) {
        pipeline->WaitPhaseError();
    }
    PROFILE_END(pipeline_wait_phase_error);

    // Clause 3.13.1 - Fraction frequency offset estimation
    PROFILE_BEGIN(calculate_phase_error);
    float average_cyclic_error = 0;
    for (auto& pipeline: pipelines) {
        const float cyclic_error = pipeline->GetAveragePhaseError();
        average_cyclic_error += cyclic_error;
    }
    average_cyclic_error /= (float)(params.nb_frame_symbols);
    // Calculate adjustments to fine frequency offset 
    const float fine_freq_error = CalculateFineFrequencyError(average_cyclic_error);
    const float beta = cfg.sync.fine_freq_update_beta;
    const float delta = -beta*fine_freq_error;
    UpdateFineFrequencyOffset(delta);
    PROFILE_END(calculate_phase_error);

    PROFILE_BEGIN(pipeline_wait_end);
    for (auto& pipeline: pipelines) {
        pipeline->WaitEnd();
    }
    PROFILE_END(pipeline_wait_end);

    total_frames_read++;
    PROFILE_BEGIN(coordinator_signal_end);
    coordinator_thread->SignalEnd();
    PROFILE_END(coordinator_signal_end);

    PROFILE_BEGIN(obs_on_ofdm_frame);
    obs_on_ofdm_frame.Notify(pipeline_out_bits);
    PROFILE_END(obs_on_ofdm_frame);
}

// Thread 3xN: Process ofdm frame
// Clause 3.14: OFDM symbol demodulator
// Clause 3.14.1: Cyclic prefix removal
// Clause 3.14.2: FFT
// Clause 3.14.3: Zero padding removal from FFT (Only include the carriers that are associated with this OFDM transmitter)
// Clause 3.15: Differential demodulator
// DOC: ETSI EN 300 401
// Referring to clause 14.5 - QPSK symbol mapper 
void OFDM_Demod::PipelineThread(OFDM_Demod_Pipeline_Thread& thread_data, OFDM_Demod_Pipeline_Thread* dependent_thread_data) {
    PROFILE_BEGIN_FUNC();

    const int symbol_start = (int)thread_data.GetSymbolStart();
    const int symbol_end = (int)thread_data.GetSymbolEnd();
    const int total_symbols = symbol_end-symbol_start;
    const int symbol_end_no_null = std::min(symbol_end, (int)params.nb_frame_symbols);
    const int symbol_end_dqpsk = std::min(symbol_end, (int)params.nb_frame_symbols-1);

    PROFILE_BEGIN(pipeline_wait_start);
    thread_data.WaitStart();
    PROFILE_END(pipeline_wait_start);

    if (thread_data.IsStopped()) {
        return;
    }

    PROFILE_BEGIN(data_processing);

    PROFILE_BEGIN(apply_pll);
    auto pipeline_time_buffer = tcb::span(active_buffer);
    auto symbols_time_buf = pipeline_time_buffer.subspan(
        symbol_start *params.nb_symbol_period, 
        total_symbols*params.nb_symbol_period);

    // Correct for frequency offset in our signal
    // NOTE: We create a local copy of the frequency offset since it
    //       can be changed in the reader thread due to coarse frequency correction
    const float frequency_offset = freq_coarse_offset + freq_fine_offset;
    const int sample_offset = symbol_start*(int)params.nb_symbol_period;
    const float dt_start = CalculateTimeOffset(sample_offset, frequency_offset);
    ApplyPLL(symbols_time_buf, symbols_time_buf, frequency_offset, dt_start); 
    PROFILE_END(apply_pll);

    // Clause 3.13: Frequency offset estimation and correction
    // Clause 3.13.1 - Fraction frequency offset estimation
    // Get phase error using cyclic prefix (ignore null symbol)
    PROFILE_BEGIN(calculate_phase_error);
    float total_phase_error = 0.0f;
    for (int i = symbol_start; i < symbol_end_no_null; i++) {
        auto sym_buf = pipeline_time_buffer.subspan(i*params.nb_symbol_period, params.nb_symbol_period);
        const float cyclic_error = CalculateCyclicPhaseError(sym_buf);
        total_phase_error += cyclic_error;
    }
    thread_data.SetAveragePhaseError(total_phase_error);
    PROFILE_END(calculate_phase_error);
    // Signal to coordinator thread that phase errors have been calculated
    PROFILE_BEGIN(pipeline_signal_phase_error);
    thread_data.SignalPhaseError();
    PROFILE_END(pipeline_signal_phase_error);

    // Clause 3.14.2 - FFT
    // Calculate fft (include null symbol)
    PROFILE_BEGIN(calculate_fft);
    for (int i = symbol_start; i < symbol_end; i++) {
        auto sym_buf = pipeline_time_buffer.subspan(i*params.nb_symbol_period, params.nb_symbol_period);
        // Clause 3.14.1 - Cyclic prefix removal
        auto data_buf = sym_buf.subspan(params.nb_cyclic_prefix, params.nb_fft);
        auto fft_buf = pipeline_fft_buffer.subspan(i*params.nb_fft, params.nb_fft);
        CalculateFFT(data_buf, fft_buf);
    }
    PROFILE_END(calculate_fft);
    // Signal to other pipeline threads which need these FFT results due to DQPSK encoding
    PROFILE_BEGIN(pipeline_signal_fft);
    thread_data.SignalFFT();
    PROFILE_END(pipeline_signal_fft);

    const auto calculate_dqpsk = [this](int start, int end) {
        const size_t nb_viterbi_bits = params.nb_data_carriers*2;

        // Clause 3.15 - Differential demodulator
        // perform our differential qpsk decoding
        for (int i = start; i < end; i++) {
            PROFILE_BEGIN(calculate_dqpsk_symbol);
            auto fft_buf_0 = pipeline_fft_buffer.subspan((i+0)*params.nb_fft, params.nb_fft);
            auto fft_buf_1 = pipeline_fft_buffer.subspan((i+1)*params.nb_fft, params.nb_fft);
            auto dqpsk_vec_buf = pipeline_dqpsk_vec_buffer.subspan(i*params.nb_data_carriers, params.nb_data_carriers);
            auto viterbi_bit_buf = pipeline_out_bits.subspan(i*nb_viterbi_bits, nb_viterbi_bits);
            CalculateDQPSK(fft_buf_1, fft_buf_0, dqpsk_vec_buf);
            CalculateViterbiBits(dqpsk_vec_buf, viterbi_bit_buf);

            PROFILE_BEGIN(calculate_viterbi_bits);
            const auto N = params.nb_data_carriers;
            PROFILE_END(calculate_viterbi_bits);
        }
    };

    // Get DQPSK result for last symbol in this thread 
    // which is dependent on other threads finishing
    if (dependent_thread_data != NULL) {
        PROFILE_BEGIN(calculate_independent_dqpsk);
        calculate_dqpsk(symbol_start, symbol_end_dqpsk-1);
        PROFILE_END(calculate_independent_dqpsk);

        PROFILE_BEGIN(dependent_pipeline_wait_fft);
        dependent_thread_data->WaitFFT();
        PROFILE_END(dependent_pipeline_wait_fft);

        PROFILE_BEGIN(calculate_dependent_dqpsk);
        calculate_dqpsk(symbol_end_dqpsk-1, symbol_end_dqpsk);
        PROFILE_END(calculate_dependent_dqpsk);
    } else {
        PROFILE_BEGIN(calculate_independent_dqpsk);
        calculate_dqpsk(symbol_start, symbol_end_dqpsk);
        PROFILE_END(calculate_independent_dqpsk);
    }

    // TODO: optional - and we need to calculate the average
    // Calculate symbol magnitude (include null symbol)
    // for (int i = symbol_start; i < symbol_end; i++) {
    //     auto* fft_buf = &pipeline_fft_buffer[i*nb_fft_length];
    //     auto* fft_mag_buf = &pipeline_fft_mag_buffer[i*nb_fft_length];
    //     CalculateMagnitude(fft_buf, fft_mag_buf);
    // }

    PROFILE_BEGIN(pipeline_signal_end);
    thread_data.SignalEnd();
    PROFILE_END(pipeline_signal_end);
}

float OFDM_Demod::ApplyPLL(
    tcb::span<const std::complex<int16_t>> x, tcb::span<std::complex<int16_t>> y, 
    const float freq_offset, const float dt0)
{
    PROFILE_BEGIN_FUNC();
    #if !USE_PLL_TABLE

    // NOTE: We don't rescale so the FFT doesn't zero out our results
    // float dt = apply_pll_scalar(x, y, freq_offset, dt0, 1);
    float dt = apply_pll_avx2(x, y, freq_offset, dt0, 1);
    return dt;

    #else

    const int N = (int)x.size();
    const int K = (int)PLL_TABLE.GetFrequencyResolution();
    const int M = (int)PLL_TABLE.GetTableSize();
    const int step = (int)freq_offset / (int)K;

    int dt = (int)(dt0);
    dt = ((dt % M) + M) % M;
    for (int i = 0; i < N; i++) {
        y[i] = x[i] * PLL_TABLE[dt];
        dt = (dt + step + M) % M;
    }
    return (float)(dt);

    #endif
}

// Since we may be breaking up the PLL calculation over multiple threads
// We need to make sure that the end of one PLL matches with the start of the next PLL
float OFDM_Demod::CalculateTimeOffset(const size_t i, const float freq_offset) {
    PROFILE_BEGIN_FUNC();
    #if !USE_PLL_TABLE
    const float dt = 2.0f * (float)M_PI * freq_offset* Ts * (float)i;
    return std::fmod(dt, 2.0f*(float)M_PI);
    #else
    const int K = (int)PLL_TABLE.GetFrequencyResolution();
    const int M = (int)PLL_TABLE.GetTableSize();
    const int step = (int)freq_offset/ (int)K;

    // (ab)modn = (amodn*bmodn)modn
    const int j = (int)i % M;
    return (float)((step*j) % M);
    #endif
}

float OFDM_Demod::CalculateCyclicPhaseError(tcb::span<const std::complex<int16_t>> sym) {
    PROFILE_BEGIN_FUNC();
    // Clause 3.13.1 - Fraction frequency offset estimation
    const size_t N = params.nb_cyclic_prefix;
    const size_t M = params.nb_fft;

    auto error_vec = std::complex<int32_t>(0,0);
    for (int i = 0; i < N; i++) {
        auto error = std::conj(sym[i] / FIXED_POINT_SCALING) * (sym[M+i] / FIXED_POINT_SCALING);
        error_vec += error;
    }

    return std::atan2((float)error_vec.imag(), (float)error_vec.real());
}

float OFDM_Demod::CalculateFineFrequencyError(const float cyclic_phase_error) {
    PROFILE_BEGIN_FUNC();
    // Clause 3.13.1 - Fraction frequency offset estimation
    // Definition of cyclic prefix
    // wd = OFDM frequency spacing = FFT bin width
    // Let w0 be a subcarrier, w0=k1*wd, k1 is an integer
    // Prefix = e^jw0(t+T), Data = e^jw0t
    // Since the prefix is equal to the data in an OFDM symbol
    // w0(t+T) = w0t + 2*k2*pi, k2 is an integer
    // T = k2*(2*pi)/w0                 (equ 1) 
    // 
    // Calculation of phase error (no frequency error)
    // phi = conj(prefix)*data
    // phi = e^-jw0(t+T) * e^jw0t
    // phi = e^-jw0T = e^(-j*k2*2*pi)
    // error = arg(phi) = -2*pi*k2 = 0
    // 
    // Calculate of phase error (with frequency offset)
    // Let w1 = frequency offset
    // Prefix = e^jw0(t+T) * e^jw1(t+T) = e^j(w0+w1)(t+T)
    // Data   = e^jw0t     * e^jw1t     = e^j(w0+w1)t
    // phi = conj(prefix)*data
    // phi = e^-j(w0+w1)(t+T) * e^j(w0+w1)t
    // phi = e^-j(w0+w1)T
    // error = arg(phi) 
    // error = (w0+w1)T
    // error = (w0+w1)/w0 * k2 * 2*pi, using (equ 1)
    // error = k2*2*pi + (w1/w0)*k2*2*pi
    // error = k2 * w1/w0 * 2*pi
    // error = w1/w0 * 2*pi,            (since |error| <= pi, then k2=1)
    // error = w1/(k1*wd) * 2*pi,       (w0=k1*wd)
    // w1 = k1 * wd/2 * error/pi       
    //
    // Since |w1| <= wd/2 due to coarse frequency correction
    // w1 = wd/2 * error/pi,            (k1=1)
    //
    // Since |error| <= pi
    // then w1 = [-wd/2, wd/2] which is our fine frequency correction range
    const float w_d = (float)(params.freq_carrier_spacing);
    const float w_error = w_d/2.0f * cyclic_phase_error/(float)(M_PI);
    return w_error;
}

// Two threads may try to update the fine frequency offset simulataneously 
// Reader thread: Runs coarse frequency correction during frame sychronisation which also affects fine frequency offset
// Coordinator thread: Joins phase errors from pipeline thread and calculates average adjustment for fine frequency offset
void OFDM_Demod::UpdateFineFrequencyOffset(const float delta) {
    PROFILE_BEGIN_FUNC();
    auto lock = std::scoped_lock(mutex_freq_fine_offset);

    const float ofdm_freq_spacing = (float)(params.freq_carrier_spacing);
    freq_fine_offset += delta;

    // NOTE: If the fine frequency adjustment is just on the edge of overflowing
    //       We add enough margin to stop this from occuring
    #if !USE_PLL_TABLE
    const float overflow_margin = 10.0f;
    #else
    const float overflow_margin = 2.0f*(float)PLL_TABLE.GetFrequencyResolution();
    #endif
    freq_fine_offset = std::fmod(freq_fine_offset, ofdm_freq_spacing/2.0f + overflow_margin);
}


void OFDM_Demod::CalculateDQPSK(
    tcb::span<const std::complex<int16_t>> in0, 
    tcb::span<const std::complex<int16_t>> in1, 
    tcb::span<std::complex<int16_t>> out_vec)
{
    PROFILE_BEGIN_FUNC();
    const int M = (int)params.nb_data_carriers/2;
    const int N_fft = (int)params.nb_fft;

    // Clause 3.14.3 - Zero padding removal
    // We store the subcarriers that carry information
    for (int i = -M, subcarrier_index = 0; i <= M; i++) {
        // The DC bin carries no information
        if (i == 0) {
            continue;
        }
        const int fft_index = (N_fft+i) % N_fft;

        // arg(z1*~z0) = arg(z1)+arg(~z0) = arg(z1)-arg(z0)
        const auto phase_delta_vec = in1[fft_index] * std::conj(in0[fft_index]);
        out_vec[subcarrier_index] = phase_delta_vec;
        subcarrier_index++;
    }
}

void OFDM_Demod::CalculateViterbiBits(tcb::span<const std::complex<int16_t>> vec_buf, tcb::span<viterbi_bit_t> bit_buf) {
    PROFILE_BEGIN_FUNC();
    const size_t N = params.nb_data_carriers;

    // DOC: ETSI EN 300 401
    // Referring to clause 14.5 - QPSK symbol mapper 
    // Deinterleave the subcarriers using carrier mapper
    // For an OFDM symbol with 2K bits, the nth symbol uses bits i and i+K
    for (int i = 0; i < N; i++) {
        const size_t j = carrier_mapper[i];
        const auto& vec = vec_buf[j];
        // const float A = std::abs(vec);
        // NOTE: Use the L1 norm since it doesn't truncate like L2 norm
        //       I.e. When real=imag, then we expect b0=A, b1=A
        //            But with L2 norm, we get b0=0.707*A, b1=0.707*A
        //                with L1 norm, we get b0=A, b1=A as expected
        const int16_t L1_mag = std::max(std::abs(vec.real()), std::abs(vec.imag()));

        // DOC: ETSI EN 300 401
        // Referring to clause 14.5 - QPSK symbol mapper
        // phi = (1-2*b0) + (1-2*b1)*1j 
        // x0 = 1-2*b0, x1 = 1-2*b1
        // b = (1-x)/2

        // NOTE: Phil Karn's viterbi decoder is configured so that b => b' : (0,1) => (-A,+A)
        // Where b is the logical bit value, and b' is the value used for soft decision decoding
        // b' = (2*b-1) * A 
        // b' = (1-x-1)*A
        // b' = -A*x

        // NOTE: We give some margin to prevent overflowing
        const int16_t A_margin = 16;
        const int16_t A = (SOFT_DECISION_VITERBI_HIGH-A_margin);
        const int16_t K = std::max((int16_t)(L1_mag/A), (int16_t)1);

        // Normalised vector has a L1 magnitude of A
        const auto norm_vec = vec / K;

        bit_buf[i]   = (viterbi_bit_t)(-norm_vec.real());
        bit_buf[i+N] = (viterbi_bit_t)(+norm_vec.imag());
    }
}

void OFDM_Demod::CalculateFFT(tcb::span<const std::complex<int16_t>> fft_in, tcb::span<std::complex<int16_t>> fft_out) {
    kiss_fft(
        fft_cfg, 
        (const kiss_fft_cpx*)fft_in.data(), 
        (kiss_fft_cpx*)fft_out.data());
}

void OFDM_Demod::CalculateIFFT(tcb::span<const std::complex<int16_t>> fft_in, tcb::span<std::complex<int16_t>> fft_out) {
    kiss_fft(
        ifft_cfg, 
        (const kiss_fft_cpx*)fft_in.data(), 
        (kiss_fft_cpx*)fft_out.data());
}

void OFDM_Demod::CalculateRelativePhase(tcb::span<const std::complex<int16_t>> fft_in, tcb::span<std::complex<int16_t>> arg_out) {
    PROFILE_BEGIN_FUNC();
    const int N = (int)params.nb_fft;
    for (int i = 0; i < (N-1); i++) {
        const auto vec = std::conj(fft_in[i]) * fft_in[i+1];
        arg_out[i] = vec;
    }
    arg_out[N-1] = {0,0};
}

void OFDM_Demod::CalculateMagnitude(tcb::span<const std::complex<int16_t>> fft_buf, tcb::span<float> mag_buf) {
    PROFILE_BEGIN_FUNC();
    const size_t N = params.nb_fft;
    const size_t M = N/2;
    for (int i = 0; i < N; i++) {
        const size_t j = (i+M) % N;
        const float x = 20.0f*std::log10((float)std::abs(fft_buf[j]));
        mag_buf[i] = x;
    }
}

float OFDM_Demod::CalculateL1Average(tcb::span<const std::complex<int16_t>> block) {
    PROFILE_BEGIN_FUNC();
    const size_t N = block.size();
    int16_t l1_avg = 0;
    for (int i = 0; i < N; i++) {
        auto& v = block[i];
        l1_avg += std::abs(v.real()) + std::abs(v.imag());
    }
    l1_avg /= (int16_t)N;
    return (float)l1_avg;
}

void OFDM_Demod::UpdateSignalAverage(tcb::span<const std::complex<int16_t>> block) {
    PROFILE_BEGIN_FUNC();
    const size_t N = block.size();
    const size_t K = (size_t)cfg.signal_l1.nb_samples;
    const size_t M = N-K;
    const size_t L = K*cfg.signal_l1.nb_decimate;
    const float beta = cfg.signal_l1.update_beta;

    for (size_t i = 0; i < M; i+=L) {
        auto* buf = &block[i];
        const float l1_avg = CalculateL1Average({buf, K});
        signal_l1_average = 
            (beta)*signal_l1_average +
            (1.0f-beta)*l1_avg;
    }
}
