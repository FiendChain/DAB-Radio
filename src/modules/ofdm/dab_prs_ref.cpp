#define _USE_MATH_DEFINES
#include <cmath>

#include "dab_prs_ref.h"
#include <stdexcept>
#include <fmt/core.h>

struct PRS_Table_Entry {
    int k_min;
    int k_max;
    int i;
    int n;
};

// DOC: ETSI EN 300 401
// Referring to clause 14.3.2 - Phase reference symbol 
// The phase reference symbol is construction using two tables
// Table 23 which contains PRS_Table_Entry, and Table 24 which contains a list of h-values

// DOC: docs/DAB_implementation_in_SDR_detailed.pdf
// For the other transmission modes including I,II,III,IV refer to appendix B
// This detailed document provides the required tables for these other transmission modes as well
static const PRS_Table_Entry PRS_PARAMS_MODE_I[] = {
    {-768, -737, 0, 1},
    {-736, -705, 1, 2},
    {-704, -673, 2, 0},
    {-672, -641, 3, 1},
    {-640, -609, 0, 3},
    {-608, -577, 1, 2},
    {-576, -545, 2, 2},
    {-544, -513, 3, 3},
    {-512, -481, 0, 2},
    {-480, -449, 1, 1},
    {-448, -417, 2, 2},
    {-416, -385, 3, 3},
    {-384, -353, 0, 1},
    {-352, -321, 1, 2},
    {-320, -289, 2, 3},
    {-288, -257, 3, 3},
    {-256, -225, 0, 2},
    {-224, -193, 1, 2},
    {-192, -161, 2, 2},
    {-160, -129, 3, 1},
    {-128,  -97, 0, 1},
    {-96,   -65, 1, 3},
    {-64,   -33, 2, 1},
    {-32,    -1, 3, 2},
    {  1,    32, 0, 3},
    { 33,    64, 3, 1},
    { 65,    96, 2, 1},
    { 97,   128, 1, 1},
    { 129,  160, 0, 2},
    { 161,  192, 3, 2},
    { 193,  224, 2, 1},
    { 225,  256, 1, 0},
    { 257,  288, 0, 2},
    { 289,  320, 3, 2},
    { 321,  352, 2, 3},
    { 353,  384, 1, 3},
    { 385,  416, 0, 0},
    { 417,  448, 3, 2},
    { 449,  480, 2, 1},
    { 481,  512, 1, 3},
    { 513,  544, 0, 3},
    { 545,  576, 3, 3},
    { 577,  608, 2, 3},
    { 609,  640, 1, 0},
    { 641,  672, 0, 3},
    { 673,  704, 3, 0},
    { 705,  736, 2, 1},
    { 737,  768, 1, 1},
};

static const PRS_Table_Entry PRS_PARAMS_MODE_II[] = {
    {-192, -161, 0, 2},
    {-160, -129, 1, 3},
    {-128, -97,  2, 2},
    {-96,  -65,  3, 2},
    {-64,  -33,  0, 1},
    {-32,  -1,   1, 2},
    {1,    32,   2, 0},
    {33,   64,   1, 2},
    {65,   96,   0, 2},
    {97,   128,  3, 1},
    {129,  160,  2, 0},
    {161,  192,  1, 3},
};

static const PRS_Table_Entry PRS_PARAMS_MODE_III[] = {
    {-96, -65, 0, 2},
    {-64, -33, 1, 3},
    {-32, -1,  2, 0},
    {1,   32,  3, 2},
    {33,  64,  2, 2},
    {65,  96,  1, 2},
};

static const PRS_Table_Entry PRS_PARAMS_MODE_IV[] = {
    {-384, -353, 0, 0},
    {-352, -321, 1, 1},
    {-320, -289, 2, 1},
    {-288, -257, 3, 2},
    {-256, -225, 0, 2},
    {-224, -193, 1, 2},
    {-192, -161, 2, 0},
    {-160, -129, 3, 3},
    {-128,  -97, 0, 3},
    {-96,   -65, 1, 1},
    {-64,   -33, 2, 3},
    {-32,    -1, 3, 2},
    {  1,    32, 0, 0},
    { 33,    64, 3, 1},
    { 65,    96, 2, 0},
    { 97,   128, 1, 2},
    { 129,  160, 0, 0},
    { 161,  192, 3, 1},
    { 193,  224, 2, 2},
    { 225,  256, 1, 2},
    { 257,  288, 0, 2},
    { 289,  320, 3, 1},
    { 321,  352, 2, 3},
    { 353,  384, 1, 0},
};

static const int H_TABLE[4][32] = {
    {0, 2, 0, 0, 0, 0, 1, 1, 2, 0, 0, 0, 2, 2, 1, 1, 0, 2, 0, 0, 0, 0, 1, 1, 2, 0, 0, 0, 2, 2, 1, 1},
    {0, 3, 2, 3, 0, 1, 3, 0, 2, 1, 2, 3, 2, 3, 3, 0, 0, 3, 2, 3, 0, 1, 3, 0, 2, 1, 2, 3, 2, 3, 3, 0},
    {0, 0, 0, 2, 0, 2, 1, 3, 2, 2, 0, 2, 2, 0, 1, 3, 0, 0, 0, 2, 0, 2, 1, 3, 2, 2, 0, 2, 2, 0, 1, 3},
    {0, 1, 2, 1, 0, 3, 3, 2, 2, 3, 2, 1, 2, 1, 3, 2, 0, 1, 2, 1, 0, 3, 3, 2, 2, 3, 2, 1, 2, 1, 3, 2},
};

static const PRS_Table_Entry* PRS_PARAMS_MODE_TABLE[4] = {
    PRS_PARAMS_MODE_I, 
    PRS_PARAMS_MODE_II, 
    PRS_PARAMS_MODE_III, 
    PRS_PARAMS_MODE_IV, 
};

inline std::complex<int16_t> GetPRS(const float phi) {
    const float FIXED_POINT_SCALING = 128.0f;
    return { 
        (int16_t)(std::cos(phi) * FIXED_POINT_SCALING),
        (int16_t)(std::sin(phi) * FIXED_POINT_SCALING)
    };
}

void get_DAB_PRS_reference(const int transmission_mode, tcb::span<std::complex<int16_t>> buf) {
    const size_t nb_fft = buf.size();
    if (transmission_mode <= 0 || transmission_mode > 4) {
        throw std::runtime_error(fmt::format("Invalid transmission mode {}", transmission_mode));
        return;
    }

    auto p_table = PRS_PARAMS_MODE_TABLE[transmission_mode-1];
    const int nb_carriers = -2*p_table[0].k_min + 1;

    if (nb_fft < nb_carriers) {
        throw std::runtime_error(fmt::format("FFT buffer not large enough to fit phase reference symbol {}<{}", 
            nb_fft, nb_carriers));
        return;
    }

    for (int i = 0; i < nb_fft; i++) {
        buf[i] = std::complex<float>(0,0);
    }

    // DOC: ETSI EN 300 401
    // Referring to clause 14.3.2 - Phase reference symbol 
    // The equation for constructing the PRS in terms of a list of phases for each subcarrier is given
    // In our demodulator code this is equivalent to the FFT result
    int p_table_index = 0;
    const int k_min = p_table[0].k_min;
    const int k_max = -k_min; 

    // -F/2 <= f < 0
    for (int k = k_min; k < 0; k++) {
        auto& e = p_table[p_table_index];
        const int h = H_TABLE[e.i][k-e.k_min];
        const float phi = (float)M_PI / 2.0f * (float)(h + e.n);
        if (k >= e.k_max) {
            p_table_index++;
        }
        buf[nb_fft+k] = GetPRS(phi);
    }

    // 0 < f <= F/2
    for (int k = 1; k <= k_max; k++) {
        auto& e = p_table[p_table_index];
        const int h = H_TABLE[e.i][k-e.k_min];
        const float phi = (float)M_PI / 2.0f * (float)(h + e.n);
        if (k >= e.k_max) {
            p_table_index++;
        }
        // NOTE: 0th bin of fft is the DC value which is 0
        buf[k] = GetPRS(phi);
    }
}