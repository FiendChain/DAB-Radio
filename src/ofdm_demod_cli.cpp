// Reads in raw IQ values from rtl_sdr and converts it into a digital OFDM frame

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <complex>
#include <assert.h>

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

#include <memory>
#include <vector>
#include <mutex>

class App 
{
private:
    FILE* fp_in;
    FILE* fp_out;
    std::mutex mutex_fp_in;
    std::mutex mutex_fp_out;
    // buffers
    std::vector<std::complex<uint8_t>> buf_rd;
    std::vector<std::complex<int16_t>> buf_rd_raw;
    // objects
    std::unique_ptr<OFDM_Demod> demod;
    // runner state
    bool is_output = true;
public:
    App(const int transmission_mode, const int total_demod_threads, FILE* const _fp_in, FILE* const _fp_out, const int _block_size) 
    : fp_in(_fp_in), fp_out(_fp_out)
    {
        buf_rd.resize(_block_size);
        buf_rd_raw.resize(_block_size);

        demod = Create_OFDM_Demodulator(transmission_mode, total_demod_threads);

        using namespace std::placeholders;
        demod->On_OFDM_Frame().Attach(std::bind(&App::OnOFDMFrame, this, _1));
        auto& cfg = demod->GetConfig();
        cfg.data_sym_mag.is_update = false;
    }
    ~App() {
        Close();
    }
    auto* GetDemod(void) { return demod.get(); }
    const auto& GetRawBuffer(void) { return buf_rd_raw; }
    auto& GetIsOutput(void) { return is_output; }
    void Run() {
        while (true) {
            const size_t block_size = buf_rd.size();
            size_t nb_read = 0;
            {
                auto lock = std::scoped_lock(mutex_fp_in);
                nb_read = fread(buf_rd.data(), sizeof(std::complex<uint8_t>), block_size, fp_in);
            }

            if (nb_read != block_size) {
                fprintf(stderr, "Failed to read data %zu/%zu\n", nb_read, block_size);
                break;
            }

            ConvertRawToExpected(buf_rd, buf_rd_raw);
            demod->Process(buf_rd_raw);
        }
    }
    void Close() {
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
    void OnOFDMFrame(tcb::span<const viterbi_bit_t> phases) {
        if (!is_output) {
            return;
        }

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

void usage() {
    fprintf(stderr, 
        "ofdm_demod_cli, runs OFDM demodulation on raw IQ values\n\n"
        "\t[-b block size (default: 8192)]\n"
        "\t[-i input filename (default: None)]\n"
        "\t    If no file is provided then stdin is used\n"
        "\t[-o output filename (default: None)]\n"
        "\t    If no file is provided then stdout is used\n"
        "\t[-M dab transmission mode (default: 1)]\n"
        "\t[-t total ofdm demod threads (default: auto)]\n"
        "\t[-D (disable output)]\n"
        "\t[-h (show usage)]\n"
    );
}

int main(int argc, char** argv) {
    int total_demod_threads = 0;
    int block_size = 8192;
    int transmission_mode = 1;
    bool is_output = true;
    char* rd_filename = NULL;
    char* wr_filename = NULL;

    int opt; 
    while ((opt = getopt_custom(argc, argv, "b:i:o:M:t:Dh")) != -1) {
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
        case 'D':
            is_output = false;
            break;
        case 'M':
            transmission_mode = (int)(atof(optarg));
            break;
        case 't':
            total_demod_threads = (int)(atof(optarg));
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
    app.GetIsOutput() = is_output;
    app.Run();

    return 0;
}