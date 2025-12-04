// src/gpu/tautomerize.cu
//
// GPU tautomer canonicalization kernel for open-standardizer.
//
// This is a *local* canonicalizer, not a full copy of
// RDKit::MolStandardize::TautomerEnumerator:
//
//   - no SMARTS transform catalog on GPU
//   - no full enumeration of tautomers
//
// Instead, it applies a set of motif-specific bond/charge tweaks
// that push the molecule toward the same kinds of canonical
// tautomers RDKit’s scorer prefers:
//
//   * nitro vs aci-nitro
//   * keto/enol + thioketo/enethiol
//   * generic C=hetero / N=O / P=O
//   * amide vs imidic / enamine-like C–N
//
// It never changes connectivity and never explicitly moves protons;
// we just:
//
//   - propagate formal charges
//   - adjust bond orders / aromatic flags locally
//
// Layout (from gpu_structs.hpp):
//   GAtom: { atomicNum, valence, degree, formalCharge, tautomerFlags, ... }
//   GBond: { idxA, idxB, bondOrder, aromaticFlag, tautomerFlags, ... }
//
// One kernel handles both atoms and bonds:
//
//   threads [0, numAtoms)     -> atom phase
//   threads [numAtoms, ...)   -> bond phase
//

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

#include "gpu_structs.hpp"         // GAtom, GBond
#include "tautomer_motif_tags.hpp" // ATF_* / BTF_* bitflags from CPU pre-pass


// =====================================================
// BASIC TYPE PREDICATES
// =====================================================

__device__ __forceinline__
bool is_carbon(uint8_t Z) {
    return Z == 6;
}

__device__ __forceinline__
bool is_oxygen(uint8_t Z) {
    return Z == 8;
}

__device__ __forceinline__
bool is_nitrogen(uint8_t Z) {
    return Z == 7;
}

__device__ __forceinline__
bool is_sulfur(uint8_t Z) {
    return Z == 16;
}

__device__ __forceinline__
bool is_phosphorus(uint8_t Z) {
    return Z == 15;
}

__device__ __forceinline__
bool is_selenium(uint8_t Z) {
    return Z == 34;
}

__device__ __forceinline__
bool is_tellurium(uint8_t Z) {
    return Z == 52;
}

// “hetero” in the tautomer-scoring sense (not H or C)
__device__ __forceinline__
bool is_hetero(uint8_t Z) {
    return (Z != 1 && Z != 6);
}

// Carbonyl-ish carbon: high-ish valence carbon
__device__ __forceinline__
bool is_carbonyl_like(const GAtom &A) {
    return (A.atomicNum == 6 && A.valence >= 3);
}

// Strong carbonyl center: we are happy to enforce C=O/C=S
__device__ __forceinline__
bool is_strong_carbonyl(const GAtom &A) {
    return (A.atomicNum == 6 && A.valence >= 3 && A.formalCharge <= 0);
}

// “oxo/thioketo-like” hetero
__device__ __forceinline__
bool is_oxo_like(uint8_t Z) {
    return (Z == 8 || Z == 16);  // O or S
}


// =====================================================
// NITRO / ACI-NITRO NORMALIZATION
// =====================================================

__device__ __forceinline__
bool is_nitro_center(const GAtom &A) {
    // N with high valence and non-negative charge
    return (A.atomicNum == 7 && A.valence >= 4 && A.formalCharge >= 0);
}

__device__ __forceinline__
bool is_nitro_oxygen(const GAtom &A) {
    // Nitro-like oxygen: low valence, often neutral or negative
    return (A.atomicNum == 8 && A.valence <= 2 && A.formalCharge <= 0);
}


// =====================================================
// CARBONYL-LIKE PROMOTION HELPERS
// =====================================================

// C–X -> C=X (X = O, S) when carbon looks like a carbonyl center
__device__ __forceinline__
bool should_promote_c_eq_oxo(const GAtom &C, const GAtom &X) {
    if (!is_strong_carbonyl(C)) return false;
    if (!is_oxo_like(X.atomicNum)) return false;
    return true;
}

// Generic C=hetero (C=O, C=N, C=P, etc.) “C=hetero” motif
__device__ __forceinline__
bool should_promote_c_eq_hetero(const GAtom &C, const GAtom &Het) {
    if (!is_carbonyl_like(C)) return false;
    if (!is_hetero(Het.atomicNum)) return false;
    return true;
}


// =====================================================
// AMIDE / IMIDIC-LIKE BEHAVIOR
// =====================================================

__device__ __forceinline__
bool should_demote_c_n_double(const GAtom &C, const GAtom &N) {
    // Prefer carbonyl C=O and keep C–N single in classic amides.
    if (!is_carbon(C.atomicNum) || !is_nitrogen(N.atomicNum)) return false;
    if (!is_carbonyl_like(C)) return false;
    // If valence is already high, double C–N is suspicious.
    return (C.valence > 3);
}


// =====================================================
// N=O / P=O / generic hetero=O motifs
// =====================================================

__device__ __forceinline__
bool is_generic_n_oxide_center(const GAtom &A) {
    // Nitrogen with at least valence 3, often neutral or positive
    return (A.atomicNum == 7 && A.valence >= 3);
}

__device__ __forceinline__
bool is_phosphoryl_center(const GAtom &A) {
    // P(V) type center
    return (A.atomicNum == 15 && A.valence >= 4);
}


// =====================================================
// MAIN KERNEL
// =====================================================

extern "C" __global__
void tautomerize_kernel(
    const GAtom *atoms,
    const GBond *bonds,
    int8_t  *outCharges,
    uint8_t *outBondOrder,
    uint8_t *outAromatic,
    int      numAtoms,
    int      numBonds
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // -------------------------------------------------
    // ATOM PHASE
    // -------------------------------------------------
    if (tid < numAtoms) {
        const GAtom A = atoms[tid];

        // Conservative: we do not move protons in this kernel.
        // Just propagate formal charges to the output.
        outCharges[tid] = A.formalCharge;
        return;
    }

    // -------------------------------------------------
    // BOND PHASE
    // -------------------------------------------------
    int bid = tid - numAtoms;
    if (bid >= numBonds) return;

    const GBond B = bonds[bid];

    const int i1 = B.idxA;
    const int i2 = B.idxB;

    const GAtom A1 = atoms[i1];
    const GAtom A2 = atoms[i2];

    // motif flags (from CPU pre-pass in tautomer_motif_tags.cpp)
    const uint16_t bondFlags = B.tautomerFlags;
    const uint16_t atf1      = A1.tautomerFlags;
    const uint16_t atf2      = A2.tautomerFlags;

    (void)atf1; // currently unused but kept for future motif rules
    (void)atf2;

    const uint8_t Z1 = A1.atomicNum;
    const uint8_t Z2 = A2.atomicNum;

    const int8_t  c1 = A1.formalCharge;
    const int8_t  c2 = A2.formalCharge;

    uint8_t order = B.bondOrder;
    uint8_t arom  = B.aromaticFlag;

    // Default: keep original unless a rule fires
    uint8_t newOrder = order;
    uint8_t newArom  = arom;

    // -------------------------------------------------
    // 0) Explicit motif-based tweaks (if you tagged bonds)
    // -------------------------------------------------

    // Example: nitro path flagged by CPU
    if (bondFlags & BTF_TAUT_NITRO_PATH) {
        bool o_neg = ((A1.atomicNum == 8 && A1.formalCharge < 0) ||
                      (A2.atomicNum == 8 && A2.formalCharge < 0));

        if (order == 1 || order == 2) {
            newOrder = o_neg ? 1 : 2;  // N–O(-) vs N=O
            newArom  = 0;
        }
    }

    // Example: keto/enol path, with keto center tagged on carbon
    if (bondFlags & BTF_TAUT_KETO_ENOL_PATH) {
        // this is a C–O / C–S bond in a keto/enol transform
        bool c_is_1 = (A1.atomicNum == 6);
        const GAtom &C = c_is_1 ? A1 : A2;
        const GAtom &X = c_is_1 ? A2 : A1;

        if (order == 1 && (C.tautomerFlags & ATF_TAUT_KETO_CENTER)) {
            (void)X; // reserved for future use
            newOrder = 2;
            newArom  = 0;
        }
    }

    // -------------------------------------------------
    // 1) Nitro / aci-nitro canonicalization
    // -------------------------------------------------
    bool n1 = is_nitro_center(A1);
    bool n2 = is_nitro_center(A2);

    if ((n1 && is_nitro_oxygen(A2)) ||
        (n2 && is_nitro_oxygen(A1))) {

        bool o_is_neg = (is_oxygen(Z1) && c1 < 0) ||
                        (is_oxygen(Z2) && c2 < 0);

        if (order == 1 || order == 2) {
            // O(-) prefers single N–O(-); neutral/positive side prefers N=O
            newOrder = o_is_neg ? 1 : 2;
            newArom  = 0;
        }
    }

    // -------------------------------------------------
    // 2) Keto / enol & thioketo normalization
    // -------------------------------------------------
    if (is_carbon(Z1) && is_oxo_like(Z2)) {
        const GAtom &C = A1;
        const GAtom &X = A2;

        if (should_promote_c_eq_oxo(C, X)) {
            if (order == 1 && c1 >= 0 && c2 <= 0) {
                newOrder = 2;
                newArom  = 0;
            }
        }
    } else if (is_carbon(Z2) && is_oxo_like(Z1)) {
        const GAtom &C = A2;
        const GAtom &X = A1;

        if (should_promote_c_eq_oxo(C, X)) {
            if (order == 1 && c2 >= 0 && c1 <= 0) {
                newOrder = 2;
                newArom  = 0;
            }
        }
    }

    // -------------------------------------------------
    // 3) Amide vs imidic / enamine-like C–N
    // -------------------------------------------------
    if (is_carbon(Z1) && is_nitrogen(Z2)) {
        const GAtom &C = A1;
        const GAtom &N = A2;

        if (order == 2 && should_demote_c_n_double(C, N)) {
            newOrder = 1;
            newArom  = 0;
        }
    } else if (is_carbon(Z2) && is_nitrogen(Z1)) {
        const GAtom &C = A2;
        const GAtom &N = A1;

        if (order == 2 && should_demote_c_n_double(C, N)) {
            newOrder = 1;
            newArom  = 0;
        }
    }

    // -------------------------------------------------
    // 4) Extended motif rules
    // -------------------------------------------------

    // 4a) Generic N=O normalization (outside explicit nitro case)
    if (newOrder == order &&
        ((is_nitrogen(Z1) && is_oxygen(Z2)) ||
         (is_nitrogen(Z2) && is_oxygen(Z1)))) {

        bool n_oxide_1 = is_generic_n_oxide_center(A1);
        bool n_oxide_2 = is_generic_n_oxide_center(A2);

        if ((n_oxide_1 || n_oxide_2) && (order == 1)) {
            newOrder = 2;
            newArom  = 0;
        }
    }

    // 4b) P=O / phosphoryl-like normalization
    if (newOrder == order &&
        ((is_phosphorus(Z1) && is_oxygen(Z2)) ||
         (is_phosphorus(Z2) && is_oxygen(Z1)))) {

        const GAtom &P = is_phosphorus(Z1) ? A1 : A2;
        const GAtom &O = is_phosphorus(Z1) ? A2 : A1;

        if (is_phosphoryl_center(P) && order == 1 && O.formalCharge <= 0) {
            newOrder = 2;
            newArom  = 0;
        }
    }

    // 4c) Generic C=hetero motif
    if (newOrder == order &&
        ((is_carbon(Z1) && is_hetero(Z2)) ||
         (is_carbon(Z2) && is_hetero(Z1)))) {

        const bool c_is_1 = is_carbon(Z1);
        const GAtom &C = c_is_1 ? A1 : A2;
        const GAtom &Het = c_is_1 ? A2 : A1;

        if (should_promote_c_eq_hetero(C, Het) && order == 1) {
            if (Het.formalCharge <= 1) {
                newOrder = 2;
                newArom  = 0;
            }
        }
    }

    // 4d) Hetero=O / hetero=hetero patterns (N=O, P=O, S=O, etc.)
    if (newOrder == order &&
        (is_oxygen(Z1) || is_oxygen(Z2))) {

        const bool o_is_1 = is_oxygen(Z1);
        const GAtom &O = o_is_1 ? A1 : A2;
        const GAtom &X = o_is_1 ? A2 : A1;

        bool oxo_center =
            (is_nitrogen(X.atomicNum) && X.valence >= 3) ||
            (is_sulfur(X.atomicNum)   && X.valence >= 4) ||
            (is_selenium(X.atomicNum) && X.valence >= 4) ||
            (is_tellurium(X.atomicNum)&& X.valence >= 4) ||
            (is_phosphorus(X.atomicNum) && X.valence >= 4);

        if (oxo_center && order == 1 && O.formalCharge <= 0) {
            newOrder = 2;
            newArom  = 0;
        }
    }

    // 4e) Rough “C(=hetero)-hetero” sanity
    if (newOrder == order &&
        ((is_carbon(Z1) && is_hetero(Z2)) ||
         (is_carbon(Z2) && is_hetero(Z1)))) {

        const bool c_is_1 = is_carbon(Z1);
        const GAtom &C = c_is_1 ? A1 : A2;
        const GAtom &Het = c_is_1 ? A2 : A1;

        if (order == 2 && C.valence > 3 && Het.formalCharge > 1) {
            newOrder = 1;
            newArom  = 0;
        }
    }

    // -------------------------------------------------
    // 5) Write back
    // -------------------------------------------------
    outBondOrder[bid] = newOrder;
    outAromatic[bid]  = newArom;
}
