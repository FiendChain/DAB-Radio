// Basic radio app that includes the OFDM demodulator and the DAB digital decoder

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <complex>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif

#include "modules/ofdm/ofdm_demodulator.h"
#include "modules/ofdm/dab_ofdm_params_ref.h"
#include "modules/ofdm/dab_prs_ref.h"
#include "modules/ofdm/dab_mapper_ref.h"
#include "modules/ofdm/ofdm_helpers.h"
#include "modules/basic_radio/basic_radio.h"

#include "audio/portaudio_output.h"
#include "audio/audio_mixer.h"
#include "audio/resampled_pcm_player.h"
#include "audio/portaudio_utility.h"

#include <unordered_map>
#include <vector>
#include <memory>
#include <thread>
#include "utility/double_buffer.h"

#include <GLFW/glfw3.h> 
#include <imgui.h>
#include <implot.h>
#include "gui/imgui_skeleton.h"
#include "gui/font_awesome_definitions.h"

#include "gui/render_ofdm_demod.h"
#include "gui/basic_radio/render_simple_view.h"
#include "gui/render_portaudio_controls.h"
#include "gui/render_profiler.h"

#include "utility/getopt/getopt.h"
#include "easylogging++.h"
#include "modules/dab/logging.h"

class App 
{
private:
    FILE* fp_in;

    std::vector<std::complex<uint8_t>> rd_in_raw;
    std::vector<std::complex<int16_t>> rd_in_data;
    std::unique_ptr<DoubleBuffer<viterbi_bit_t>> frame_double_buffer;

    std::unique_ptr<OFDM_Demod> ofdm_demod;
    std::unique_ptr<BasicRadio> radio;
    std::unique_ptr<SimpleViewController> radio_view_controller;

    std::unique_ptr<std::thread> ofdm_demod_thread;
    std::unique_ptr<std::thread> basic_radio_thread;

    PaDeviceList pa_devices;
    PortAudio_Output pa_output;
    std::unordered_map<subchannel_id_t, std::unique_ptr<Resampled_PCM_Player>> dab_plus_audio_players;
public:
    App(const int transmission_mode, const int total_demod_threads, FILE* const _fp_in, const int _block_size)
    : fp_in(_fp_in) 
    {
        auto params = get_dab_parameters(transmission_mode);

        rd_in_raw.resize(_block_size);
        rd_in_data.resize(_block_size);
        frame_double_buffer = std::make_unique<DoubleBuffer<viterbi_bit_t>>(params.nb_frame_bits);

        ofdm_demod = Create_OFDM_Demodulator(transmission_mode, total_demod_threads);
        radio = std::make_unique<BasicRadio>(params);
        radio_view_controller = std::make_unique<SimpleViewController>(*(radio.get()));

        using namespace std::placeholders;
        radio->On_DAB_Plus_Channel().Attach(std::bind(&App::Attach_DAB_Plus_Audio_Player, this, _1, _2));
        ofdm_demod->On_OFDM_Frame().Attach(std::bind(&App::OnOFDMFrame, this, _1));

        ofdm_demod_thread = std::make_unique<std::thread>([this]() {
            RunnerThread_OFDM_Demod();
        });
        basic_radio_thread = std::make_unique<std::thread>([this]() {
            RunnerThread_Radio();
        });

        #ifdef _WIN32
        const auto target_host_api_index = Pa_HostApiTypeIdToHostApiIndex(PORTAUDIO_TARGET_HOST_API_ID);
        const auto target_device_index = Pa_GetHostApiInfo(target_host_api_index)->defaultOutputDevice;
        pa_output.Open(target_device_index);
        #else
        pa_output.Open(Pa_GetDefaultOutputDevice());
        #endif
    }
    ~App() {
        fclose(fp_in);
        fp_in = NULL;
        frame_double_buffer->Close();
        ofdm_demod_thread->join();
        basic_radio_thread->join();
    }
    const auto& GetSourceBuffer() { return rd_in_data; }
    auto& GetOFDMDemod() { return *(ofdm_demod.get()); }
    auto& GetRadio() { return *(radio.get()); }
    auto& GetViewController() { return *(radio_view_controller.get()); }
    auto& GetPaAudioOutput() { return pa_output; }
    auto& GetPaDevices() { return pa_devices; }
private:
    // ofdm thread -> ofdm frame callback -> double buffer -> dab thread
    void RunnerThread_OFDM_Demod() {
        // Read raw 8bit IQ values and convert them to floating point
        while (true) {
            if (fp_in == NULL) return;

            const size_t block_size = rd_in_raw.size();
            const size_t nb_read = fread(rd_in_raw.data(), sizeof(std::complex<uint8_t>), block_size, fp_in);
            if (nb_read != block_size) {
                fprintf(stderr, "Failed to read in %zu bytes, got %zu bytes\n", block_size, nb_read);
                break;
            }

            ConvertRawToExpected(rd_in_raw, rd_in_data);
            ofdm_demod->Process(rd_in_data);
        }
    }    
    void OnOFDMFrame(tcb::span<const viterbi_bit_t> buf) {
        auto* inactive_buf = frame_double_buffer->AcquireInactiveBuffer();
        if (inactive_buf == NULL) return;

        const size_t nb_frame_bits = buf.size();
        for (int i = 0; i < nb_frame_bits; i++) {
            inactive_buf[i] = buf[i];
        }
        frame_double_buffer->ReleaseInactiveBuffer();
    }
    void RunnerThread_Radio() {
        while (true) {
            auto* active_buf = frame_double_buffer->AcquireActiveBuffer();
            if (active_buf == NULL) return;

            const size_t nb_frame_bits = frame_double_buffer->GetLength();
            radio->Process({active_buf, nb_frame_bits});
            frame_double_buffer->ReleaseActiveBuffer();
        }
    }
    // Create audio player for each subchannel
    void Attach_DAB_Plus_Audio_Player(subchannel_id_t subchannel_id, Basic_DAB_Plus_Channel& channel) {
		auto& controls = channel.GetControls();

        auto& mixer = pa_output.GetMixer();
        auto buf = mixer.CreateManagedBuffer(2);

        auto player = std::make_unique<Resampled_PCM_Player>(buf, pa_output.GetSampleRate());
		auto res = dab_plus_audio_players.emplace(subchannel_id, std::move(player)).first;
		auto* pcm_player = res->second.get(); 
		channel.OnAudioData().Attach([this, &controls, pcm_player](BasicAudioParams params, tcb::span<const uint8_t> buf) {
			if (!controls.GetIsPlayAudio()) {
				return;
			}

            pcm_player->SetInputSampleRate(params.frequency);

            tcb::span<const Frame<int16_t>> rd_buf = {
                reinterpret_cast<const Frame<int16_t>*>(buf.data()),
                (size_t)(buf.size() / sizeof(Frame<int16_t>))
            };

			pcm_player->ConsumeBuffer(rd_buf);
		});
	}
};

class Renderer: public ImguiSkeleton 
{
private:
    App& app;
public:
    Renderer(App& _app): app(_app) {}
    virtual GLFWwindow* Create_GLFW_Window(void) {
        return glfwCreateWindow(
            1280, 720, 
            "Complete DAB Radio", 
            NULL, NULL);
    }
    virtual void AfterImguiContextInit() {
        ImPlot::CreateContext();
        ImguiSkeleton::AfterImguiContextInit();

        auto& io = ImGui::GetIO();
        io.IniFilename =  "imgui_basic_radio.ini";
        io.Fonts->AddFontFromFileTTF("res/Roboto-Regular.ttf", 15.0f);
        {
            static const ImWchar icons_ranges[] = { ICON_MIN_FA, ICON_MAX_FA };
            ImFontConfig icons_config;
            icons_config.MergeMode = true;
            icons_config.PixelSnapH = true;
            io.Fonts->AddFontFromFileTTF("res/font_awesome.ttf", 16.0f, &icons_config, icons_ranges);
        }
        ImGuiSetupCustomConfig();
    }
    virtual void Render() {
		RenderProfiler();

        if (ImGui::Begin("Demodulator")) {
            ImGuiID dockspace_id = ImGui::GetID("Demodulator Dockspace");
            ImGui::DockSpace(dockspace_id);

            auto& buf = app.GetSourceBuffer();
            RenderSourceBuffer(buf);
            RenderOFDMDemodulator(app.GetOFDMDemod());
        }
        ImGui::End();

        if (ImGui::Begin("Simple View")) {
            ImGuiID dockspace_id = ImGui::GetID("Simple View Dockspace");
            ImGui::DockSpace(dockspace_id);
            if (ImGui::Begin("Audio Controls")) {
                RenderPortAudioControls(app.GetPaDevices(), app.GetPaAudioOutput());
            }
            ImGui::End();
            RenderSimple_Root(app.GetRadio(), app.GetViewController());
        }
        ImGui::End();
    }
    virtual void AfterShutdown() {
        ImPlot::DestroyContext();
    }
};

void usage() {
    fprintf(stderr, 
        "basic_radio_app, Complete radio app with demodulator and dab decoding\n\n"
        "\t[-i input filename (default: None)]\n"
        "\t    If no file is provided then stdin is used\n"
        "\t[-v Enable logging (default: false)]\n"
        "\t[-b block size (default: 8192)]\n"
        "\t[-M dab transmission mode (default: 1)]\n"
        "\t[-t total ofdm demod threads (default: auto)]\n"
        "\t[-h (show usage)]\n"
    );
}

INITIALIZE_EASYLOGGINGPP
int main(int argc, char** argv) {
    int total_demod_threads = 0;
    char* rd_filename = NULL;
    int block_size = 8192;
    bool is_logging = false;
    int transmission_mode = 1;

    int opt; 
    while ((opt = getopt_custom(argc, argv, "i:b:M:t:vh")) != -1) {
        switch (opt) {
        case 'i':
            rd_filename = optarg;
            break;
        case 'b':
            block_size = (int)(atof(optarg));
            break;
        case 'M':
            transmission_mode = (int)(atof(optarg));
            break;
        case 't':
            total_demod_threads = (int)(atof(optarg));
            break;
        case 'v':
            is_logging = true;
            break;
        case 'h':
        default:
            usage();
            return 0;
        }
    }
    
    if (block_size <= 0) {
        fprintf(stderr, "Block size must be positive (%d)\n", block_size); 
        return 1;

    }

    if (transmission_mode <= 0 || transmission_mode > 4) {
        fprintf(stderr, "Transmission modes: I,II,III,IV are supported not (%d)\n", transmission_mode);
        return 1;
    }

    // app startup
    FILE* fp_in = stdin;
    if (rd_filename != NULL) {
        fp_in = fopen(rd_filename, "rb");
        if (fp_in == NULL) {
            fprintf(stderr, "Failed to open file for reading\n");
            return 1;
        }
    }

#ifdef _WIN32
    _setmode(_fileno(fp_in), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif

    auto dab_loggers = RegisterLogging();
    auto basic_radio_logger = el::Loggers::getLogger("basic-radio");

    el::Configurations defaultConf;
    const char* logging_level = is_logging ? "true" : "false";
    defaultConf.setToDefault();
    defaultConf.setGlobally(el::ConfigurationType::Enabled, logging_level);
    defaultConf.setGlobally(el::ConfigurationType::Format, "[%level] [%thread] [%logger] %msg");
    el::Loggers::reconfigureAllLoggers(defaultConf);
    el::Helpers::setThreadName("main-thread");

    auto port_audio_handler = ScopedPaHandler();
    auto app = App(transmission_mode, total_demod_threads, fp_in, block_size);
    auto renderer = Renderer(app);
    const int rv = RenderImguiSkeleton(&renderer);
    return rv;
}

