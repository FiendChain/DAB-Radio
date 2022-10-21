#pragma once
#include <stdint.h>

#include "database/dab_database_entities.h"

class CIF_Deinterleaver;
class Trellis;
class ViterbiDecoder;
class AdditiveScrambler;

// Is associated with a subchannel residing inside the CIF (common interleaved frame)
// Performs deinterleaving and decoding on that subchannel
class MSC_Decoder 
{
private:
    const Subchannel subchannel;
    // Internal buffers
    const int nb_encoded_bits;
    const int nb_encoded_bytes;
    uint8_t* encoded_bits_buf;
    uint8_t* decoded_bits_buf;
    uint8_t* decoded_bytes_buf;
    // Decoders and deinterleavers
    CIF_Deinterleaver* deinterleaver;
    Trellis* trellis;
    ViterbiDecoder* vitdec;
    AdditiveScrambler* scrambler;
public:
    MSC_Decoder(const Subchannel _subchannel);
    ~MSC_Decoder();
    // Returns the number of bytes decoded
    int DecodeCIF(const uint8_t* buf, const int N);
    const uint8_t* GetDecodedBytes() { return decoded_bytes_buf; }
private:
    int DecodeEEP();
    int DecodeUEP();
};