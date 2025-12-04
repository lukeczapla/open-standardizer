// src/gpu/mesomerizer.cu
//
// Resonance / mesomerization normalization kernel.
// This is *not* a full resonance enumerator, just a local
// "smoother" that:
//   - relaxes localized negative charge (phenoxide/enolate-like)
//   - normalizes nitro bond orders
//   - regularizes carboxylate C=O / C–O(−)
//   - flattens conjugated systems toward single/double, non-aromatic.
//
// Layout: GAtom / GBond from gpu_structs.hpp.

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

#include "gpu_structs.hpp"
#include "gpu_kernels.hpp"

// =====================================================
// FAST DEVICE HELPERS
// =====================================================

// conjugated double bond participants
__device__ __forceinline__
bool is_conjugatable(uint8_t Z, uint8_t val, uint8_t degree) {
    // sp2 carbons, heteroatoms with lone pair, etc.
    if (Z == 6 && val == 3) return true;          // C(sp2)
    if ((Z == 7 || Z == 8) && val <= 2) return true; // N/O with lone-pair-ish valence
    return false;
}

// phenoxide & enolate charge shift
__device__ __forceinline__
bool can_shift_negative(uint8_t Z, int8_t charge) {
    return (charge == -1 && (Z == 6 || Z == 7 || Z == 8));
}

// nitro canonicalization helpers
__device__ __forceinline__
bool is_nitro_center(uint8_t Z, uint8_t val, int8_t charge) {
    return (Z == 7 && val == 4 && charge >= 0);
}

__device__ __forceinline__
bool is_nitro_oxygen(uint8_t Z, uint8_t val, int8_t charge) {
    return (Z == 8 && val == 1 && charge <= 0);
}

// carboxylate O− ↔ C=O flipping
__device__ __forceinline__
bool is_carboxylate_pair(
    uint8_t Z1, uint8_t Z2,
    uint8_t val1, uint8_t val2,
    int8_t  c1,   int8_t  c2
) {
    // Loose C(sp2)=O / C–O(−) recognition
    return (Z1 == 6 && Z2 == 8 && val1 == 3 && val2 <= 2 && (c2 == -1 || c1 == 0));
}

// =====================================================
// MESOMERIZATION KERNEL
// (1 pass: distribute & normalize resonance patterns)
// =====================================================

extern "C" __global__
void mesomerizer_kernel(
    const GAtom *atoms,
    const GBond *bonds,
    int8_t  *outCharges,
    uint8_t *outBondOrder,
    uint8_t *outAromatic,
    int      numAtoms,
    int      numBonds
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // ------------------------------------------------
    // ATOM PHASE
    // Each thread handles an atom-level resonance shift
    // ------------------------------------------------
    if (tid < numAtoms) {
        const GAtom A = atoms[tid];
        int8_t c = A.formalCharge;

        // 1. Phenoxide/enolate charge migration:
        //    O− or C− → move negative charge into conjugation
        if (can_shift_negative(A.atomicNum, c)) {
            // Slightly relax charge to mimic delocalization.
            // This is intentionally conservative: we don't
            // move charge across atoms, just soften it.
            int8_t adj = (abs(c) > 0 ? c / 2 : 0);
            outCharges[tid] = adj;
        } else {
            outCharges[tid] = c;
        }

        return;
    }

    // ------------------------------------------------
    // BOND PHASE
    // ------------------------------------------------
    int bid = tid - numAtoms;
    if (bid >= numBonds) return;

    const GBond B = bonds[bid];

    // NOTE: current GBond uses idxA / idxB, not a1 / a2
    const GAtom A1 = atoms[B.idxA];
    const GAtom A2 = atoms[B.idxB];

    const uint8_t Z1   = A1.atomicNum;
    const uint8_t Z2   = A2.atomicNum;

    const int8_t  c1   = A1.formalCharge;
    const int8_t  c2   = A2.formalCharge;

    const uint8_t val1 = A1.valence;
    const uint8_t val2 = A2.valence;

    uint8_t order = B.bondOrder;
    uint8_t arom  = B.aromaticFlag;

    // Default: no change
    uint8_t newOrder = order;
    uint8_t newArom  = arom;

    // -----------------------------
    // 1. Nitro mesomerization
    //    N(+)(=O)-O(−) ↔ N(−)(O=O) → normalize to
    //    a reasonable N=O / N–O(−) pattern.
    // -----------------------------
    if (is_nitro_center(Z1, val1, c1) && is_nitro_oxygen(Z2, val2, c2)) {
        // canonical nitro: N=O and N–O(−)
        if (val2 == 1)
            newOrder = 1;  // N–O(−)
        else
            newOrder = 2;  // N=O

        newArom = 0;
        outBondOrder[bid] = newOrder;
        outAromatic[bid]  = newArom;
        return;
    }
    if (is_nitro_center(Z2, val2, c2) && is_nitro_oxygen(Z1, val1, c1)) {
        if (val1 == 1)
            newOrder = 1;
        else
            newOrder = 2;

        newArom = 0;
        outBondOrder[bid] = newOrder;
        outAromatic[bid]  = newArom;
        return;
    }

    // -----------------------------
    // 2. Carboxylate resonance
    //    O− ↔ C=O flipping
    // -----------------------------
    if (is_carboxylate_pair(Z1, Z2, val1, val2, c1, c2)) {
        if (c2 == -1) {
            newOrder = 1;   // C–O(−)
        } else {
            newOrder = 2;   // C=O
        }
        newArom = 0;

        outBondOrder[bid] = newOrder;
        outAromatic[bid]  = newArom;
        return;
    }
    if (is_carboxylate_pair(Z2, Z1, val2, val1, c2, c1)) {
        if (c1 == -1) {
            newOrder = 1;
        } else {
            newOrder = 2;
        }
        newArom = 0;

        outBondOrder[bid] = newOrder;
        outAromatic[bid]  = newArom;
        return;
    }

    // -----------------------------
    // 3. Conjugation leveling
    //    If both atoms conjugatable → push toward
    //    a stable single/double, non-aromatic bond.
    // -----------------------------
    if (is_conjugatable(Z1, val1, A1.degree) &&
        is_conjugatable(Z2, val2, A2.degree)) {

        // Remove aromatic flag in ambiguous conjugated systems
        newArom = 0;

        // Soften bond: move "toward" single/double average
        if (order == 1)
            newOrder = 1;  // stable single
        else if (order >= 2)
            newOrder = 2;  // stable double
        else
            newOrder = order;

        outBondOrder[bid] = newOrder;
        outAromatic[bid]  = newArom;
        return;
    }

    // -----------------------------
    // 4. Default: no change
    // -----------------------------
    outBondOrder[bid] = order;
    outAromatic[bid]  = arom;
}
