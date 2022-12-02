// Basic radio scraper that includes the OFDM demodulator and the DAB digital decoder
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
#include "modules/basic_scraper/basic_scraper.h"

#include <thread>
#include <memory>
#include "utility/double_buffer.h"

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
    std::unique_ptr<BasicScraper> scraper;

    std::unique_ptr<std::thread> ofdm_demod_thread;
    std::unique_ptr<std::thread> basic_radio_thread;
public:
    App(const int transmission_mode, const int total_demod_threads, FILE* const _fp_in, const int _block_size, const char* dir)
    : fp_in(_fp_in)
    {
        auto params = get_dab_parameters(transmission_mode);

        rd_in_raw.resize(_block_size);
        rd_in_data.resize(_block_size);
        frame_double_buffer = std::make_unique<DoubleBuffer<viterbi_bit_t>>(params.nb_frame_bits);

        radio = std::make_unique<BasicRadio>(params);
        scraper = std::make_unique<BasicScraper>(*(radio.get()), dir);
        ofdm_demod = Create_OFDM_Demodulator(transmission_mode, total_demod_threads);

        using namespace std::placeholders;
        ofdm_demod->On_OFDM_Frame().Attach(std::bind(&App::OnOFDMFrame, this, _1));

        ofdm_demod_thread = std::make_unique<std::thread>([this]() {
            RunnerThread_OFDM_Demod();
        });

        basic_radio_thread = std::make_unique<std::thread>([this]() {
            RunnerThread_Radio();
        });
    }
    ~App() {
        // ofdm_demod_thread only joins when we have finished reading all data
        ofdm_demod_thread->join();
        frame_double_buffer->Close();
        basic_radio_thread->join();
    }
private:
    // ofdm thread -> ofdm frame callback -> double buffer -> dab thread
    void RunnerThread_OFDM_Demod() {
        while (true) {
            if (fp_in == NULL) return;

            const int block_size = (int)rd_in_raw.size();
            const auto nb_read = (int)fread(rd_in_raw.data(), sizeof(std::complex<uint8_t>), block_size, fp_in);
            if (nb_read != block_size) {
                fprintf(stderr, "Failed to read in %d bytes, got %d bytes\n", block_size, nb_read);
                break;
            }

            ConvertRawToExpected(rd_in_raw, rd_in_data);
            ofdm_demod->Process(rd_in_data);
        }
    }    
    void OnOFDMFrame(tcb::span<const viterbi_bit_t> buf) {
        auto* inactive_buf = frame_double_buffer->AcquireInactiveBuffer();
        if (inactive_buf == NULL) {
            return;
        }
        const size_t nb_frame_bits = frame_double_buffer->GetLength();
        for (int i = 0; i < nb_frame_bits; i++) {
            inactive_buf[i] = buf[i];
        }
        frame_double_buffer->ReleaseInactiveBuffer();
    }
    void RunnerThread_Radio() {
        while (true) {
            auto* active_buf = frame_double_buffer->AcquireActiveBuffer();
            if (active_buf == NULL) {
                return;
            }
            const size_t nb_frame_bits = frame_double_buffer->GetLength();
            radio->Process({active_buf, nb_frame_bits});
            frame_double_buffer->ReleaseActiveBuffer();
        }
    }
};

void usage() {
    fprintf(stderr, 
        "basic_radio_scraper, Demodulates signal and saves DAB channel data\n\n"
        "\t[-o output directory (default: scraper_out)]\n"
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
    const char* output_dir = NULL;
    const char* rd_filename = NULL;
    int total_demod_threads = 0;
    int block_size = 8192;
    bool is_logging = false;
    int transmission_mode = 1;

    int opt; 
    while ((opt = getopt_custom(argc, argv, "o:i:b:M:t:vh")) != -1) {
        switch (opt) {
        case 'o':
            output_dir = optarg;
            break;
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

    if (output_dir == NULL) {
        output_dir = "scraper_out";
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
    auto basic_scraper_logger = el::Loggers::getLogger("basic-scraper");

    el::Configurations defaultConf;
    const char* logging_level = is_logging ? "true" : "false";
    defaultConf.setToDefault();
    defaultConf.setGlobally(el::ConfigurationType::Enabled, logging_level);
    defaultConf.setGlobally(el::ConfigurationType::Format, "[%level] [%thread] [%logger] %msg");
    el::Loggers::reconfigureAllLoggers(defaultConf);
    el::Helpers::setThreadName("main-thread");

    el::Configurations scraper_conf; 
    scraper_conf.setToDefault();
    scraper_conf.setGlobally(el::ConfigurationType::Enabled, "true");
    scraper_conf.setGlobally(el::ConfigurationType::Format, "[%level] [%thread] [%logger] %msg");
    basic_scraper_logger->configure(scraper_conf);

    fprintf(stderr, "Writing to directory %s\n", output_dir);
    auto app = App(transmission_mode, total_demod_threads, fp_in, block_size, output_dir);
    return 0;
}

