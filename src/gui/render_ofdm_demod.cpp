#include "render_ofdm_demod.h"
#include "modules/ofdm/ofdm_demodulator.h"

#include "imgui.h"
#include "implot.h"

void RenderSourceBuffer(tcb::span<const std::complex<int16_t>> buf_raw)
{
    const size_t block_size = buf_raw.size();
    static double x = 0.0f;
    if (ImGui::Begin("Sampling buffer")) {
        if (ImPlot::BeginPlot("Block")) {
            const auto* buf = reinterpret_cast<const int16_t*>(buf_raw.data());
            ImPlot::SetupAxisLimits(ImAxis_Y1, -128, 128, ImPlotCond_Once);
            ImPlot::PlotLine("Real", &buf[0], (int)block_size, 1.0f, 0, 0, 0, 2*sizeof(buf[0]));
            ImPlot::PlotLine("Imag", &buf[1], (int)block_size, 1.0f, 0, 0, 0, 2*sizeof(buf[0]));
            ImPlot::EndPlot();
        }
    }
    ImGui::End();
}

void RenderOFDMDemodulator_Plots(OFDM_Demod& demod);
void RenderOFDMDemodulator_Controls(OFDM_Demod& demod);
void RenderOFDMDemodulator_State(OFDM_Demod& demod);

void RenderOFDMDemodulator(OFDM_Demod& demod) {
    RenderOFDMDemodulator_State(demod);
    RenderOFDMDemodulator_Controls(demod);
    RenderOFDMDemodulator_Plots(demod);
}

void RenderOFDMDemodulator_Plots(OFDM_Demod& demod) {
    const auto params = demod.GetOFDMParams();
    auto& cfg = demod.GetConfig();

    const int total_symbols = (int)params.nb_frame_symbols;
    const int total_dqpsk_symbols = total_symbols-1;
    static int symbol_index = 0;

    if (ImGui::Begin("Raw Signal")) {
        ImGui::SliderInt("DQPSK Symbol Index", &symbol_index, 0, total_dqpsk_symbols-1);

        const size_t N = params.nb_data_carriers;
        auto syms_vec_data = demod.GetFrameDataVec();
        auto sym_vec = syms_vec_data.subspan(symbol_index*N, N);

        if (ImPlot::BeginPlot("Raw constellation", ImVec2(-1,0), ImPlotFlags_Equal)) {
            const double A = 40000;
            ImPlot::SetupAxisLimits(ImAxis_X1, -A, A, ImPlotCond_Once);
            ImPlot::SetupAxisLimits(ImAxis_Y1, -A, A, ImPlotCond_Once);

            auto* buf = reinterpret_cast<int16_t*>(sym_vec.data());
            const float marker_size = 2.0f;
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Cross, marker_size);
            ImPlot::PlotScatter("IQ", &buf[0], &buf[1], (int)N, 0, 0, 2*sizeof(buf[0]));
            ImPlot::EndPlot();
        }
    }
    ImGui::End();

    if (ImGui::Begin("Constellation")) {
        ImGui::SliderInt("DQPSK Symbol Index", &symbol_index, 0, total_dqpsk_symbols-1);

        const size_t nb_data_carriers = params.nb_data_carriers;
        const size_t nb_sym_bits = nb_data_carriers*2;
        auto syms_bits_data = demod.GetFrameDataBits();
        auto sym_bits = syms_bits_data.subspan(symbol_index*nb_sym_bits, nb_sym_bits);

        static const int NB_REFERENCE = 4;
        static const std::complex<viterbi_bit_t> REFERENCE_CONSTELLATION[NB_REFERENCE] = {
            { SOFT_DECISION_VITERBI_LOW , SOFT_DECISION_VITERBI_LOW  },
            { SOFT_DECISION_VITERBI_LOW , SOFT_DECISION_VITERBI_HIGH },
            { SOFT_DECISION_VITERBI_HIGH, SOFT_DECISION_VITERBI_LOW  },
            { SOFT_DECISION_VITERBI_HIGH, SOFT_DECISION_VITERBI_HIGH },
        };

        if (ImPlot::BeginPlot("Viterbi bits constellation", ImVec2(-1,0), ImPlotFlags_Equal)) {
            const auto A = (double)SOFT_DECISION_VITERBI_HIGH * 4.0f;
            ImPlot::SetupAxisLimits(ImAxis_X1, -A, A, ImPlotCond_Once);
            ImPlot::SetupAxisLimits(ImAxis_Y1, -A, A, ImPlotCond_Once);

            auto* buf = sym_bits.data();
            auto* ref_buf = reinterpret_cast<const viterbi_bit_t*>(REFERENCE_CONSTELLATION);
            const float marker_size = 2.0f;
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Cross, marker_size);
            ImPlot::PlotScatter("IQ", &buf[0], &buf[nb_data_carriers], (int)nb_data_carriers, 0, 0, sizeof(buf[0]));
            ImPlot::PlotScatter("Reference", &ref_buf[0], &ref_buf[1], NB_REFERENCE, 0, 0, 2*sizeof(ref_buf[0]));
            ImPlot::EndPlot();
        }
    }
    ImGui::End();

    if (ImGui::Begin("Fine time synchronisation")) {
        if (ImPlot::BeginPlot("Fine time response")) {
            auto buf = demod.GetImpulseResponse();
            ImPlot::SetupAxisLimits(ImAxis_Y1, 60, 150, ImPlotCond_Once);
            ImPlot::PlotLine("Impulse response", buf.data(), (int)buf.size());
            // Plot useful markers for fine time sync using time correlation
            int marker_id = 0;
            const int target_peak_x = (int)params.nb_cyclic_prefix;
            const int actual_peak_x = target_peak_x + demod.GetFineTimeOffset();
            double marker_0 = target_peak_x;
            double marker_1 = actual_peak_x;
            ImPlot::DragLineX(marker_id++, &marker_0, ImVec4(0,1,0,1), 1.0f, ImPlotDragToolFlags_NoInputs);
            ImPlot::DragLineX(marker_id++, &marker_1, ImVec4(1,0,0,1), 1.0f, ImPlotDragToolFlags_NoInputs);
            ImPlot::EndPlot();
        }
    }
    ImGui::End();

    if (ImGui::Begin("Coarse frequency response")) {
        static bool show_markers = true;
        if (ImPlot::BeginPlot("Coarse frequency response")) {
            auto buf = demod.GetCoarseFrequencyResponse();
            ImPlot::SetupAxisLimits(ImAxis_Y1, 180, 260, ImPlotCond_Once);
            ImPlot::PlotLine("Impulse response", buf.data(), (int)buf.size());

            // Plot useful markers for coarse freq sync using freq correlation
            const int coarse_freq_offset = (int)demod.GetCoarseFrequencyOffset();
            const int max_coarse_freq_offset = cfg.sync.max_coarse_freq_correction;
            const int freq_fft_bin = (int)params.freq_carrier_spacing;
            const int peak_offset_x = -coarse_freq_offset / freq_fft_bin;
            const int max_offset_x = max_coarse_freq_offset / freq_fft_bin;

            const auto target_colour = ImVec4(0,0.8,0,1);
            const auto actual_colour = ImVec4(1,0,0,1);
            const auto limits_colour = ImVec4(0,0,0.8,1);

            int marker_id = 0;
            const int target_peak_x = (int)params.nb_fft/2;
            const int actual_peak_x = target_peak_x + peak_offset_x;
            double marker_0 = target_peak_x;
            double marker_1 = target_peak_x - max_offset_x;
            double marker_2 = target_peak_x + max_offset_x;
            double marker_3 = actual_peak_x;
            ImPlot::DragLineX(marker_id++, &marker_0, target_colour, 1.0f, ImPlotDragToolFlags_NoInputs);
            ImPlot::DragLineX(marker_id++, &marker_1, limits_colour, 1.0f, ImPlotDragToolFlags_NoInputs);
            ImPlot::DragLineX(marker_id++, &marker_2, limits_colour, 1.0f, ImPlotDragToolFlags_NoInputs);
            ImPlot::DragLineX(marker_id++, &marker_3, actual_colour, 1.0f, ImPlotDragToolFlags_NoInputs);
            ImPlot::EndPlot();
        }
    }
    ImGui::End();

    if (ImGui::Begin("Correlation time buffer")) {
        auto buf_raw = demod.GetCorrelationTimeBuffer();
        const size_t N = buf_raw.size();
        if (ImPlot::BeginPlot("NULL+PRS")) {
            const auto* buf = reinterpret_cast<const int16_t*>(buf_raw.data());
            ImPlot::SetupAxisLimits(ImAxis_Y1, -128, 128, ImPlotCond_Once);
            ImPlot::PlotLine("Real", &buf[0], (int)N, 1.0f, 0, 0, 0, 2*sizeof(buf[0]));
            ImPlot::PlotLine("Imag", &buf[1], (int)N, 1.0f, 0, 0, 0, 2*sizeof(buf[0]));

            const auto target_colour = ImVec4(0,0.8,0,1);
            int marker_id = 0;
            const int target_x = (int)params.nb_null_period;
            double marker_0 = target_x;
            ImPlot::DragLineX(marker_id++, &marker_0, target_colour, 1.0f, ImPlotDragToolFlags_NoInputs);
            ImPlot::EndPlot();
        }
    }
    ImGui::End();

    // TODO:
    // {
    //     ImGui::Begin("Null symbol spectrum");
    //     if (ImPlot::BeginPlot("Null symbol")) {
    //         auto buf = demod.GetNullSymbolMagnitude();
    //         const int N = params.nb_fft;
    //         ImPlot::SetupAxisLimits(ImAxis_Y1, 20, 90, ImPlotCond_Once);
    //         ImPlot::PlotLine("Null symbol", buf, N);
    //         ImPlot::EndPlot();
    //     }
    //     ImGui::End();
    // }

    // TODO:
    // {
    //     ImGui::Begin("Data symbol spectrum");
    //     if (ImPlot::BeginPlot("Data symbol spectrum")) {
    //         auto buf = demod.GetFrameMagnitudeAverage();
    //         const int N = params.nb_fft;
    //         ImPlot::SetupAxisLimits(ImAxis_Y1, 20, 90, ImPlotCond_Once);
    //         ImPlot::PlotLine("Data symbol", buf, N);
    //         ImPlot::EndPlot();
    //     }
    //     ImGui::End();
    // }
}

void RenderOFDMDemodulator_Controls(OFDM_Demod& demod) {
    auto& cfg = demod.GetConfig();
    auto params = demod.GetOFDMParams();

    if (ImGui::Begin("Controls")) {
        if (ImGui::Button("Reset")) {
            demod.Reset();
        }

        ImGui::Checkbox("Update data symbol mag", &cfg.data_sym_mag.is_update);
        ImGui::SameLine();
        ImGui::Checkbox("Update tii symbol mag", &cfg.is_update_tii_sym_mag);
        ImGui::SameLine();
        ImGui::Checkbox("Coarse frequency correction", &cfg.sync.is_coarse_freq_correction);

        ImGui::SliderFloat("Fine frequency beta", &cfg.sync.fine_freq_update_beta, 0.0f, 1.0f, "%.2f");
        if (ImGui::SliderInt("Max coarse frequency (Hz)", &cfg.sync.max_coarse_freq_correction, 0, 100000)) {
            const int N = cfg.sync.max_coarse_freq_correction / (int)params.freq_carrier_spacing;
            cfg.sync.max_coarse_freq_correction = N * (int)params.freq_carrier_spacing;
        }
        ImGui::SliderFloat("Coarse freq slow beta", &cfg.sync.coarse_freq_slow_beta, 0.0f, 1.0f);

        ImGui::SliderFloat("Impulse peak threshold (dB)", &cfg.sync.impulse_peak_threshold_db, 0, 100.0f, "%.f");
        ImGui::SliderFloat("Impulse peak distance weight", &cfg.sync.impulse_peak_distance_probability, 0.0f, 1.0f, "%.3f");

        static float null_threshold[2] = {0,0};
        null_threshold[0] = cfg.null_l1_search.thresh_null_start;
        null_threshold[1] = cfg.null_l1_search.thresh_null_end;
        if (ImGui::SliderFloat2("Null detection threshold", null_threshold, 0.0f, 1.0f, "%.2f")) {
            null_threshold[0] = std::min(null_threshold[0], null_threshold[1]);
            null_threshold[1] = std::max(null_threshold[0], null_threshold[1]);
            cfg.null_l1_search.thresh_null_start = null_threshold[0];
            cfg.null_l1_search.thresh_null_end = null_threshold[1];
        }

        ImGui::SliderFloat("Data sym mag update beta", &cfg.data_sym_mag.update_beta, 0.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("L1 signal update beta", &cfg.signal_l1.update_beta, 0.0f, 1.0f, "%.2f");
    }
    ImGui::End();
}

void RenderOFDMDemodulator_State(OFDM_Demod& demod) {
    #define ENUM_TO_STRING(NAME) \
    case OFDM_Demod::State::NAME: ImGui::Text("State: "#NAME); break;

    if (ImGui::Begin("Stats")) {
        switch (demod.GetState()) {
        ENUM_TO_STRING(FINDING_NULL_POWER_DIP);
        ENUM_TO_STRING(READING_NULL_AND_PRS);
        ENUM_TO_STRING(RUNNING_COARSE_FREQ_SYNC);
        ENUM_TO_STRING(RUNNING_FINE_TIME_SYNC);
        ENUM_TO_STRING(READING_SYMBOLS);
        default: 
        ImGui::Text("State: Unknown"); 
            break;
        }
        ImGui::Text("Fine freq: %.2f Hz", demod.GetFineFrequencyOffset());
        ImGui::Text("Coarse freq: %.2f Hz", demod.GetCoarseFrequencyOffset());
        ImGui::Text("Net freq: %.2f Hz", demod.GetNetFrequencyOffset());
        ImGui::Text("Signal level: %.2f", demod.GetSignalAverage());
        ImGui::Text("Frames read: %d", demod.GetTotalFramesRead());
        ImGui::Text("Frames desynced: %d", demod.GetTotalFramesDesync());
    }
    ImGui::End();

    #undef ENUM_TO_STRING
}