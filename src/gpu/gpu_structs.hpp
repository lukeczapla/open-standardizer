// src/gpu/gpu_structs.hpp

#pragma once
#include <stdint.h>

// Output fixed stereo flags
struct StereoFixResult {
    uint8_t atomChiral;
    uint8_t bondStereo;
};

// Per-atom bit flags for tautomer-related roles
enum AtomTautomerFlags : uint16_t {
    ATF_NONE              = 0,
    ATF_TAUT_KETO_CENTER  = 1 << 0,  // carbonyl C in keto/thioketo
    ATF_TAUT_ENOL_O       = 1 << 1,  // O in enol / enethiol
    ATF_TAUT_NITRO_N      = 1 << 2,  // N in nitro / aci-nitro
    ATF_TAUT_NITRO_O      = 1 << 3,  // O in nitro / aci-nitro
    ATF_TAUT_P_CENTER     = 1 << 4,  // P center for P=O/P–OH motifs
    ATF_TAUT_HETERO_HOT   = 1 << 5,  // “mobile-H” hetero (O/N/S…)
    ATF_TAUT_N_OXIDE      = 1 << 6,  // N-oxide center
    // room for more
};

// Per-bond bit flags for tautomer paths
enum BondTautomerFlags : uint16_t {
    BTF_NONE                 = 0,
    BTF_TAUT_KETO_ENOL_PATH  = 1 << 0,  // part of keto/enol (or thioketo) transform
    BTF_TAUT_NITRO_PATH      = 1 << 1,  // nitro <-> aci-nitro
    BTF_TAUT_N_OXIDE_PATH    = 1 << 2,  // generic N=O / N-oxide
    BTF_TAUT_P_O_PATH        = 1 << 3,  // P=O / P–OH patterns
    BTF_TAUT_C_EQ_HETERO     = 1 << 4,  // “C=hetero” scoring motif
    BTF_TAUT_CEQHETERO_HET   = 1 << 5,  // “C(=hetero)-hetero” motif
    // room for more
};

struct GAtom {
    uint8_t  atomicNum;
    int8_t   formalCharge;
    uint8_t  degree;
    uint8_t  valence;
    uint8_t  hCount;        // total num Hs
    uint8_t  chiralFlag;    // 0/1/2/3 like your stereo kernel
    uint16_t tautomerFlags; // AtomTautomerFlags
};

struct GBond {
    uint16_t idxA;          // begin atom
    uint16_t idxB;          // end atom
    uint8_t  bondOrder;     // 1, 2, 3 (from BondTypeAsDouble)
    uint8_t  aromaticFlag;  // 0/1
    uint8_t  stereo;        // your stereo kernel uses this (0,3=E,4=Z)
    uint8_t  _pad;          // padding if you want (optional)
    uint16_t tautomerFlags; // BondTautomerFlags
};
