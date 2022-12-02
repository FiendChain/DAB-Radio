// Reads in raw IQ values from rtl_sdr and converts it into a digital OFDM frame

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <thread>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif

#include "utility/getopt/getopt.h"

#include "modules/ofdm/ofdm_demodulator.h"
#include "modules/ofdm/dab_ofdm_params_ref.h"
#include "modules/ofdm/dab_prs_ref.h"
#include "modules/ofdm/dab_mapper_ref.h"
#include "modules/ofdm/ofdm_helpers.h"

#include <GLFW/glfw3.h> 
#include "implot.h"
#include "gui/render_ofdm_demod.h"
#include "gui/imgui_skeleton.h"
#include "gui/font_awesome_definitions.h"

#include <complex>
#include <memory>
#include <vector>
#include <mutex>

#include "gui/render_profiler.h"

class App 
{
private:
    // buffers
    FILE* fp_in;
    FILE* fp_out;
    std::mutex mutex_fp_in;
    std::mutex mutex_fp_out;

    std::vector<std::complex<uint8_t>> buf_rd;
    std::vector<std::complex<int16_t>> buf_rd_raw;
    // objects
    std::unique_ptr<OFDM_Demod> demod;
    // runner state
    bool is_running = true;
    bool flag_step = false;
    bool flag_dump_frame = false;
    // runner thread
    std::unique_ptr<std::thread> runner_thread;
    // External controls
    bool is_wait_step = false;
    bool is_always_dump_frame = false;
public:
    App(const int transmission_mode, const int total_demod_threads, FILE* const _fp_in, FILE* const _fp_out, const int _block_size) 
    : fp_in(_fp_in), fp_out(_fp_out)
    {
        buf_rd.resize(_block_size);
        buf_rd_raw.resize(_block_size);

        demod = Create_OFDM_Demodulator(transmission_mode, total_demod_threads);

        using namespace std::placeholders;
        demod->On_OFDM_Frame().Attach(std::bind(&App::OnOFDMFrame, this, _1));

        is_running = false;
        runner_thread = NULL;
    }
    ~App() {
        Close();
        if (runner_thread->joinable()) {
            runner_thread->join();
        }
    }
    void Start() {
        if (is_running) return;
        is_running = true;
        runner_thread = std::make_unique<std::thread>([this]() {
            RunnerThread();
        });
    }
    auto& GetDemod(void) { return *(demod.get()); }
    const auto& GetRawBuffer(void) { return buf_rd_raw; }
    auto& GetIsWaitStep() { return is_wait_step; }
    auto& GetIsDumpFrame() { return is_always_dump_frame; }
    void TriggerStep() { flag_step = true; }
    void TriggerDumpFrame()  { flag_dump_frame = true; }
    void Close() {
        is_running = false;
        is_wait_step = false;
        if (fp_in != NULL) {
            fclose(fp_in);
        }
        if (fp_out != NULL) {
            fclose(fp_out);
        }

        auto lock_fp_in = std::scoped_lock(mutex_fp_in);
        auto lock_fp_out = std::scoped_lock(mutex_fp_out);
        fp_in = NULL;
        fp_out = NULL;
    }
private:
    void RunnerThread() {
        while (is_running) {
            while (!(flag_step) && (is_wait_step)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
            }
            flag_step = false;

            const size_t block_size = buf_rd.size();
            size_t nb_read = 0;
            {
                auto lock = std::scoped_lock(mutex_fp_in);
                if (fp_in == NULL) {
                    is_running = false;
                    break;
                }
                nb_read = fread(buf_rd.data(), sizeof(std::complex<uint8_t>), block_size, fp_in);
            }

            if (nb_read != block_size) {
                fprintf(stderr, "Failed to read in data %zu/%zu\n", nb_read, block_size);
                break;
            }

            ConvertRawToExpected(buf_rd, buf_rd_raw);
            demod->Process(buf_rd_raw);
        }
    }
    void OnOFDMFrame(tcb::span<const viterbi_bit_t> phases) {
        if (!flag_dump_frame && !is_always_dump_frame) {
            return;
        }
        flag_dump_frame = false;

        const size_t N = phases.size();
        size_t nb_write = 0;
        {
            auto lock = std::scoped_lock(mutex_fp_out);
            if (fp_out == NULL) {
                return;
            }
            nb_write = fwrite(phases.data(), sizeof(viterbi_bit_t), N, fp_out);
        }

        if (nb_write != N) {
            fprintf(stderr, "Failed to write ofdm frame %zu/%zu\n", nb_write, N);
            Close();
        }
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
            "OFDM Demodulator GUI", 
            NULL, NULL);
    }
    virtual void AfterImguiContextInit() {
        ImPlot::CreateContext();
        ImguiSkeleton::AfterImguiContextInit();
        auto& io = ImGui::GetIO();
        io.IniFilename =  "imgui_ofdm_demod.ini";
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
            ImGui::DockSpace(ImGui::GetID("OFDM Demod Dockspace"));
            auto& buf = app.GetRawBuffer();
            RenderSourceBuffer(buf);
            RenderOFDMDemodulator(app.GetDemod());
            RenderAppControls();
        }
        ImGui::End();
    }
    virtual void AfterShutdown() {
        ImPlot::DestroyContext();
    }
private:
    void RenderAppControls() {
        if (ImGui::Begin("Input controls")) {
            ImGui::Checkbox("Enable stepping", &app.GetIsWaitStep());
            if (app.GetIsWaitStep()) {
                if (ImGui::Button("Step")) {
                    app.TriggerStep();
                }
            }             

            ImGui::Checkbox("Enable continuous frame dump", &app.GetIsDumpFrame());
            if (!app.GetIsDumpFrame()) {
                if (ImGui::Button("Dump next block")) {
                    app.TriggerDumpFrame();
                }
            } 
        }
        ImGui::End();
    }
};


void usage() {
    fprintf(stderr, 
        "ofdm_demod_gui, runs OFDM demodulation on raw IQ values with GUI\n\n"
        "\t[-b block size (default: 8192)]\n"
        "\t[-i input filename (default: None)]\n"
        "\t    If no file is provided then stdin is used\n"
        "\t[-o output filename (default: None)]\n"
        "\t    If no file is provided then stdout is used\n"
        "\t[-M dab transmission mode (default: 1)]\n"
        "\t[-t total ofdm demod threads (default: auto)]\n"
        "\t[-S toggle step mode (default: false)]\n"
        "\t[-D toggle frame output (default: true)]\n"
        "\t[-h (show usage)]\n"
    );
}

int main(int argc, char** argv) {
    int total_demod_threads = 0;
    int block_size = 8192;
    int transmission_mode = 1;
    bool is_step_mode = false;
    bool is_frame_output = true;

    char* rd_filename = NULL;
    char* wr_filename = NULL;

    int opt; 
    while ((opt = getopt_custom(argc, argv, "b:i:o:M:t:SDh")) != -1) {
        switch (opt) {
        case 'b':
            block_size = (int)(atof(optarg));
            break;
        case 'i':
            rd_filename = optarg;
            break;
        case 'o':
            wr_filename = optarg;
            break;
        case 'M':
            transmission_mode = (int)(atof(optarg));
            break;
        case 't':
            total_demod_threads = (int)(atof(optarg));
            break;
        case 'S':
            is_step_mode = true;
            break;
        case 'D':
            is_frame_output = false;
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

    FILE* fp_in = stdin;
    if (rd_filename != NULL) {
        fp_in = fopen(rd_filename, "rb");
        if (fp_in == NULL) {
            fprintf(stderr, "Failed to open file for reading\n");
            return 1;
        }
    }

    FILE* fp_out = stdout;
    if (wr_filename != NULL) {
        fp_out = fopen(wr_filename, "wb+");
        if (fp_out == NULL) {
            fprintf(stderr, "Failed to open file for writing\n");
            return 1;
        }
    }

#ifdef _WIN32
    _setmode(_fileno(fp_in), _O_BINARY);
    _setmode(_fileno(fp_out), _O_BINARY);
#endif

    auto app = App(transmission_mode, total_demod_threads, fp_in, fp_out, block_size);
    auto renderer = Renderer(app);

    app.GetIsWaitStep() = is_step_mode;
    app.GetIsDumpFrame() = is_frame_output;
    app.Start();

    const int rv = RenderImguiSkeleton(&renderer);
    return rv;
}
