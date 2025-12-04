// src/gpu/bond_infer.cu
//
// ChemAxon-like bond order normalization / inference.
// This operates *locally* on each bond using GAtom / GBond,
// never changing connectivity, only bond order + aromatic flag.

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

#include "gpu_structs.hpp"
#include "gpu_kernels.hpp"

// ======================================================
// DEVICE-LEVEL HELPERS (ChemAxon-like bond normalization)
// ======================================================

// Amide carbonyl C–N pattern
__device__ __forceinline__
bool is_amide_CN(
    uint8_t Z1, uint8_t Z2,
    uint8_t val1, uint8_t val2,
    uint8_t deg1, uint8_t deg2
) {
    // Very broad "C=O–N" amide-ish shortcut:
    //  C: Z=6, valence ~3, degree ~3
    //  N: Z=7, valence ~3
    bool carbonylCarbon =
        (Z1 == 6 && val1 == 3 && deg1 == 3);

    bool amideNitrogen =
        (Z2 == 7 && val2 == 3);

    return carbonylCarbon && amideNitrogen;
}

// Nitro group N(+)(=O)-O− form
__device__ __forceinline__
bool is_nitro(
    uint8_t Z1, uint8_t Z2,
    uint8_t val1, uint8_t val2,
    int8_t  c1,  int8_t  c2
) {
    // Broad ChemAxon-like heuristic:
    //  - N with valence 4 (often formal +1)
    //  - O with valence 1 (often O−)
    bool n_plus  = (Z1 == 7 && val1 == 4 && c1 >= 0);
    bool o_minus = (Z2 == 8 && val2 == 1 && c2 <= 0);
    return n_plus && o_minus;
}

// Carboxylate C=O / C–O− (normalize to ideal)
__device__ __forceinline__
bool is_carboxylate_pair(
    uint8_t Z1, uint8_t Z2,
    uint8_t val1, uint8_t val2,
    int8_t  c1,  int8_t  c2
) {
    // looser "C(=O)-O" / "C(–O−)" recognition
    bool carbonyl =
        (Z1 == 6 && val1 == 3);

    bool oxygen =
        (Z2 == 8 && val2 <= 2);

    return carbonyl && oxygen;
}

// ======================================================
// MAIN CUDA BOND NORMALIZATION KERNEL (1 thread per bond)
// ======================================================

extern "C" __global__
void bond_infer_kernel(
    const GBond *bonds,
    const GAtom *atoms,
    uint8_t *outBondOrder,
    uint8_t *outAromatic,
    int       numBonds
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numBonds) return;

    const GBond B  = bonds[tid];

    // NOTE: in the current GAtom / GBond layout we use idxA / idxB,
    // not a1 / a2:
    const GAtom A1 = atoms[B.idxA];
    const GAtom A2 = atoms[B.idxB];

    const uint8_t Z1   = A1.atomicNum;
    const uint8_t Z2   = A2.atomicNum;
    const uint8_t val1 = A1.valence;
    const uint8_t val2 = A2.valence;
    const uint8_t deg1 = A1.degree;
    const uint8_t deg2 = A2.degree;

    const int8_t  c1   = A1.formalCharge;
    const int8_t  c2   = A2.formalCharge;

    const uint8_t order = B.bondOrder;
    const uint8_t arom  = B.aromaticFlag;

    // =============================================
    // 1. Aromaticity override
    //
    // In the original version this looked at per-atom "aromatic"
    // flags; in our current structs we only have per-bond
    // aromaticFlag. If the bond is already marked aromatic,
    // we pin it as aromatic single for normalization.
    // =============================================
    if (arom) {
        outBondOrder[tid] = 1;  // encode aromatic as order=1 + aromaticFlag=1
        outAromatic[tid]  = 1;
        return;
    }

    // =============================================
    // 2. Amide normalization
    //    C=O–N → keep C=O and ensure C–N is single
    // =============================================
    if (is_amide_CN(Z1, Z2, val1, val2, deg1, deg2)) {
        outBondOrder[tid] = 1; // C–N single (amide)
        outAromatic[tid]  = 0;
        return;
    }
    if (is_amide_CN(Z2, Z1, val2, val1, deg2, deg1)) {
        outBondOrder[tid] = 1;
        outAromatic[tid]  = 0;
        return;
    }

    // =============================================
    // 3. Nitro normalization
    //    N(+)(=O)-O(−) → disambiguate N=O vs N–O(−)
    // =============================================
    if (is_nitro(Z1, Z2, val1, val2, c1, c2)) {
        // For valence-1 O we interpret as N–O(−); otherwise N=O
        outBondOrder[tid] = (val2 == 1 ? 1 : 2);
        outAromatic[tid]  = 0;
        return;
    }
    if (is_nitro(Z2, Z1, val2, val1, c2, c1)) {
        outBondOrder[tid] = (val1 == 1 ? 1 : 2);
        outAromatic[tid]  = 0;
        return;
    }

    // =============================================
    // 4. Carboxylate / carbonyl heuristics
    //    C=O–O− → ensure one C=O and one C–O(−)
    // =============================================
    if (is_carboxylate_pair(Z1, Z2, val1, val2, c1, c2)) {
        if (c2 == -1) {
            outBondOrder[tid] = 1;   // C–O(−)
        } else {
            outBondOrder[tid] = 2;   // C=O
        }
        outAromatic[tid] = 0;
        return;
    }

    if (is_carboxylate_pair(Z2, Z1, val2, val1, c2, c1)) {
        if (c1 == -1) {
            outBondOrder[tid] = 1;
        } else {
            outBondOrder[tid] = 2;
        }
        outAromatic[tid] = 0;
        return;
    }

    // =============================================
    // 5. Fallback: keep existing bond order & arom
    // =============================================
    outBondOrder[tid] = order;
    outAromatic[tid]  = arom;
}
