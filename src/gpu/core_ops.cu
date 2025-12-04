#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

#include "gpu_structs.hpp"
#include "core_ops_flags.hpp"
#include "core_ops_batch.hpp"
#include "gpu_kernels.hpp"

#include <GraphMol/ROMol.h>
#include <GraphMol/RWMol.h>
#include <GraphMol/Atom.h>
#include <GraphMol/Bond.h>
#include <GraphMol/MolOps.h>

using RDKit::ROMol;
// Layout assumptions (simplified):
//   - N molecules in batch
//   - total_atoms = sum_i num_atoms[i]
//   - atom_offset[m] = starting index of mol m in atom arrays
//
// Arrays:
//   atom_isotope[total_atoms]      : int (0 = none)
//   atom_flags[total_atoms]        : bitfield for stereo/H, etc.
//   atom_frag_id[total_atoms]      : connected component id per atom
//   atom_keep_mask[total_atoms]    : 1 = keep, 0 = drop (for removeFragment/removeH)
//
// We'll run one block per molecule, threads over atoms.

__global__
void clear_isotopes_kernel(
    int  const* __restrict__ mol_atom_offset,
    int  const* __restrict__ mol_num_atoms,
    int         num_mols,
    int*        atom_isotope  // in-place
) {
    int mol_id = blockIdx.x;
    if (mol_id >= num_mols) return;

    int start = mol_atom_offset[mol_id];
    int n     = mol_num_atoms[mol_id];

    int tid = threadIdx.x;
    for (int i = tid; i < n; i += blockDim.x) {
        int idx = start + i;
        atom_isotope[idx] = 0;  // clear isotopic label
    }
}


__global__
void clear_stereo_kernel(
    int  const* __restrict__ mol_atom_offset,
    int  const* __restrict__ mol_num_atoms,
    int         num_mols,
    uint32_t*   atom_flags  // in-place bitfield
) {
    int mol_id = blockIdx.x;
    if (mol_id >= num_mols) return;

    int start = mol_atom_offset[mol_id];
    int n     = mol_num_atoms[mol_id];

    int tid = threadIdx.x;
    for (int i = tid; i < n; i += blockDim.x) {
        int idx = start + i;
        uint32_t f = atom_flags[idx];
        // zero stereo bits only
        f &= ~FLAG_STEREO_MASK;
        atom_flags[idx] = f;
    }
}


// Simple per-fragment atom counting. Assumes:
//   frag_id in [0, max_frags) per molecule (pre-normalized on CPU)
__global__
void largest_fragment_kernel(
    int  const* __restrict__ mol_atom_offset,
    int  const* __restrict__ mol_num_atoms,
    int  const* __restrict__ atom_frag_id,
    int         num_mols,
    int         max_frags_per_mol,
    int*        mol_largest_frag_id    // out: per-mol largest fragment id
) {
    extern __shared__ int s_counts[];  // size: max_frags_per_mol
    int mol_id = blockIdx.x;
    if (mol_id >= num_mols) return;

    int tid = threadIdx.x;

    // init shared counts
    for (int f = tid; f < max_frags_per_mol; f += blockDim.x) {
        s_counts[f] = 0;
    }
    __syncthreads();

    int start = mol_atom_offset[mol_id];
    int n     = mol_num_atoms[mol_id];

    // count atoms per frag
    for (int i = tid; i < n; i += blockDim.x) {
        int idx = start + i;
        int frag = atom_frag_id[idx];
        if (frag >= 0 && frag < max_frags_per_mol) {
            atomicAdd(&s_counts[frag], 1);
        }
    }
    __syncthreads();

    // find largest fragment by count (thread 0)
    int best_frag = 0;
    if (tid == 0) {
        int best_count = -1;
        for (int f = 0; f < max_frags_per_mol; ++f) {
            int c = s_counts[f];
            if (c > best_count) {
                best_count = c;
                best_frag = f;
            }
        }
        mol_largest_frag_id[mol_id] = best_frag;
    }
}


__global__
void apply_largest_fragment_mask_kernel(
    int  const* __restrict__ mol_atom_offset,
    int  const* __restrict__ mol_num_atoms,
    int  const* __restrict__ atom_frag_id,
    int  const* __restrict__ mol_largest_frag_id,
    int         num_mols,
    uint8_t*    atom_keep_mask  // 1 = keep, 0 = drop
) {
    int mol_id = blockIdx.x;
    if (mol_id >= num_mols) return;

    int start = mol_atom_offset[mol_id];
    int n     = mol_num_atoms[mol_id];
    int largest_frag = mol_largest_frag_id[mol_id];

    int tid = threadIdx.x;
    for (int i = tid; i < n; i += blockDim.x) {
        int idx = start + i;
        int frag = atom_frag_id[idx];
        atom_keep_mask[idx] = (frag == largest_frag) ? 1 : 0;
    }
}

__global__
void mark_explicit_h_kernel(
    int  const* __restrict__ mol_atom_offset,
    int  const* __restrict__ mol_num_atoms,
    int         num_mols,
    uint32_t*   atom_flags,
    uint8_t*    atom_keep_mask  // 1 = keep, 0 = drop
) {
    int mol_id = blockIdx.x;
    if (mol_id >= num_mols) return;

    int start = mol_atom_offset[mol_id];
    int n     = mol_num_atoms[mol_id];
    int tid   = threadIdx.x;

    for (int i = tid; i < n; i += blockDim.x) {
        int idx = start + i;
        uint32_t f = atom_flags[idx];

        // If this bit marks explicit H (simplified example),
        // set keep_mask=0 so CPU removes this atom.
        if (f & FLAG_EXPLICIT_H) {
            atom_keep_mask[idx] = 0;
        }
    }
}

// ---------------------------------------------------------------------
// Helpers to apply atom_keep_mask back to an RDKit ROMol
// ---------------------------------------------------------------------

static ROMol build_submol_from_keep_mask(
    const ROMol &mol,
    const CoreOpsBatch &batch,
    int molIndex
) {
    using RDKit::RWMol;

    int start = batch.mol_atom_offset[molIndex];
    int n     = batch.mol_num_atoms[molIndex];

    RWMol rw;
    std::vector<int> oldToNew(n, -1);

    // 1) Add kept atoms
    for (int i = 0; i < n; ++i) {
        int idx = start + i;
        if (batch.atom_keep_mask[idx] == 0) {
            continue;
        }
        const auto* atom = mol.getAtomWithIdx(i);
        auto* newAtom = new RDKit::Atom(*atom);  // RWMol takes ownership
        int newIdx = rw.addAtom(newAtom, false, true);
        oldToNew[i] = newIdx;
    }

    // 2) Add bonds between kept atoms
    for (auto bond : mol.bonds()) {
        int a1 = bond->getBeginAtomIdx();
        int a2 = bond->getEndAtomIdx();

        if (a1 < 0 || a1 >= n || a2 < 0 || a2 >= n) {
            continue;
        }
        int na1 = oldToNew[a1];
        int na2 = oldToNew[a2];
        if (na1 < 0 || na2 < 0) {
            continue;  // at least one atom was dropped
        }

        rw.addBond(na1, na2, bond->getBondType());
        auto* newBond = rw.getBondBetweenAtoms(na1, na2);
        if (newBond) {
            newBond->setIsAromatic(bond->getIsAromatic());
            newBond->setStereo(bond->getStereo());
        }
    }

    // 3) Sanitize
    try {
        RDKit::MolOps::sanitizeMol(rw);
    } catch (...) {
        // If sanitize fails, we still return the raw RWMol as ROMol
    }

    return ROMol(rw);
}


// Allocate and upload the basic per-mol arrays used by core_ops kernels.
struct CoreOpsDeviceBuffers {
    int num_mols;
    int total_atoms;

    int *d_mol_atom_offset = nullptr;
    int *d_mol_num_atoms   = nullptr;
    int *d_atom_isotope    = nullptr;
    std::uint32_t *d_atom_flags = nullptr;
    int *d_atom_frag_id    = nullptr;
    std::uint8_t *d_atom_keep_mask = nullptr;
    int *d_mol_largest_frag_id = nullptr; // only used for largest fragment

    CoreOpsDeviceBuffers(const CoreOpsBatch &batch, bool need_frag, bool need_flags)
        : num_mols(batch.num_mols),
          total_atoms(batch.total_atoms)
    {
        cudaMalloc(&d_mol_atom_offset, num_mols * sizeof(int));
        cudaMalloc(&d_mol_num_atoms,   num_mols * sizeof(int));

        cudaMemcpy(d_mol_atom_offset, batch.mol_atom_offset.data(),
                   num_mols * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mol_num_atoms,   batch.mol_num_atoms.data(),
                   num_mols * sizeof(int), cudaMemcpyHostToDevice);

        if (!batch.atom_isotope.empty()) {
            cudaMalloc(&d_atom_isotope, total_atoms * sizeof(int));
            cudaMemcpy(d_atom_isotope, batch.atom_isotope.data(),
                       total_atoms * sizeof(int), cudaMemcpyHostToDevice);
        }

        if (need_flags && !batch.atom_flags.empty()) {
            cudaMalloc(&d_atom_flags, total_atoms * sizeof(std::uint32_t));
            cudaMemcpy(d_atom_flags, batch.atom_flags.data(),
                       total_atoms * sizeof(std::uint32_t), cudaMemcpyHostToDevice);
        }

        if (need_frag && !batch.atom_frag_id.empty()) {
            cudaMalloc(&d_atom_frag_id, total_atoms * sizeof(int));
            cudaMemcpy(d_atom_frag_id, batch.atom_frag_id.data(),
                       total_atoms * sizeof(int), cudaMemcpyHostToDevice);

            cudaMalloc(&d_mol_largest_frag_id, num_mols * sizeof(int));
        }

        // keep mask
        if (!batch.atom_keep_mask.empty()) {
            cudaMalloc(&d_atom_keep_mask, total_atoms * sizeof(std::uint8_t));
            cudaMemcpy(d_atom_keep_mask, batch.atom_keep_mask.data(),
                       total_atoms * sizeof(std::uint8_t), cudaMemcpyHostToDevice);
        }
    }

    ~CoreOpsDeviceBuffers() {
        if (d_mol_atom_offset)     cudaFree(d_mol_atom_offset);
        if (d_mol_num_atoms)       cudaFree(d_mol_num_atoms);
        if (d_atom_isotope)        cudaFree(d_atom_isotope);
        if (d_atom_flags)          cudaFree(d_atom_flags);
        if (d_atom_frag_id)        cudaFree(d_atom_frag_id);
        if (d_atom_keep_mask)      cudaFree(d_atom_keep_mask);
        if (d_mol_largest_frag_id) cudaFree(d_mol_largest_frag_id);
    }
};


// ---------------------------------------------------------------------
// gpu_kernel_clear_isotopes
//   - Build CoreOpsBatch for a single ROMol
//   - Run clear_isotopes_kernel on GPU
//   - Copy isotopes back and apply to a new ROMol
// ---------------------------------------------------------------------
ROMol gpu_kernel_clear_isotopes(const ROMol &mol) {
    std::vector<const ROMol*> mols = { &mol };
    CoreOpsBatch batch = build_core_ops_batch_from_mols(mols);

    CoreOpsDeviceBuffers dev(batch, /*need_frag=*/false, /*need_flags=*/false);

    dim3 blocks(batch.num_mols);
    dim3 threads(128);

    clear_isotopes_kernel<<<blocks, threads>>>(
        dev.d_mol_atom_offset,
        dev.d_mol_num_atoms,
        batch.num_mols,
        dev.d_atom_isotope
    );
    cudaDeviceSynchronize();

    // Download isotopes
    cudaMemcpy(batch.atom_isotope.data(), dev.d_atom_isotope,
               batch.total_atoms * sizeof(int), cudaMemcpyDeviceToHost);

    ROMol out(mol);
    int start = batch.mol_atom_offset[0];
    int n     = batch.mol_num_atoms[0];
    for (int i = 0; i < n; ++i) {
        int idx = start + i;
        auto* atom = out.getAtomWithIdx(i);
        atom->setIsotope(batch.atom_isotope[idx]);  // now 0
    }

    return out;
}

// ---------------------------------------------------------------------
// gpu_kernel_keep_largest_fragment
//   - Build CoreOpsBatch for a single ROMol
//   - Run largest_fragment_kernel + apply_largest_fragment_mask_kernel
//   - Copy atom_keep_mask back and build a new ROMol with only the
//     largest fragment.
// ---------------------------------------------------------------------
ROMol gpu_kernel_keep_largest_fragment(const ROMol &mol) {
    std::vector<const ROMol*> mols = { &mol };
    CoreOpsBatch batch = build_core_ops_batch_from_mols(mols);

    CoreOpsDeviceBuffers dev(batch, /*need_frag=*/true, /*need_flags=*/false);

    dim3 blocks(batch.num_mols);
    dim3 threads(128);
    size_t shared = static_cast<size_t>(batch.max_frags_per_mol) * sizeof(int);

    // Step 1: determines largest fragment id per molecule
    largest_fragment_kernel<<<blocks, threads, shared>>>(
        dev.d_mol_atom_offset,
        dev.d_mol_num_atoms,
        dev.d_atom_frag_id,
        batch.num_mols,
        batch.max_frags_per_mol,
        dev.d_mol_largest_frag_id
    );
    cudaDeviceSynchronize();

    // Step 2: fill atom_keep_mask (1 for largest fragment, 0 otherwise)
    apply_largest_fragment_mask_kernel<<<blocks, threads>>>(
        dev.d_mol_atom_offset,
        dev.d_mol_num_atoms,
        dev.d_atom_frag_id,
        dev.d_mol_largest_frag_id,
        batch.num_mols,
        dev.d_atom_keep_mask
    );
    cudaDeviceSynchronize();

    // Download keep mask
    cudaMemcpy(batch.atom_keep_mask.data(), dev.d_atom_keep_mask,
               batch.total_atoms * sizeof(std::uint8_t), cudaMemcpyDeviceToHost);

    // Build submol that keeps only the largest fragment
    return build_submol_from_keep_mask(mol, batch, /*molIndex=*/0);
}


// ---------------------------------------------------------------------
// gpu_kernel_remove_explicit_h
//   - Build CoreOpsBatch for a single ROMol
//   - atom_flags already has FLAG_EXPLICIT_H set for H atoms
//   - Run mark_explicit_h_kernel to set atom_keep_mask=0 for explicit H
//   - Copy atom_keep_mask back and build a new ROMol without those Hs
// ---------------------------------------------------------------------
ROMol gpu_kernel_remove_explicit_h(const ROMol &mol) {
    std::vector<const ROMol*> mols = { &mol };
    CoreOpsBatch batch = build_core_ops_batch_from_mols(mols);

    CoreOpsDeviceBuffers dev(batch, /*need_frag=*/false, /*need_flags=*/true);

    dim3 blocks(batch.num_mols);
    dim3 threads(128);

    mark_explicit_h_kernel<<<blocks, threads>>>(
        dev.d_mol_atom_offset,
        dev.d_mol_num_atoms,
        batch.num_mols,
        dev.d_atom_flags,
        dev.d_atom_keep_mask
    );
    cudaDeviceSynchronize();

    // Download keep mask
    cudaMemcpy(batch.atom_keep_mask.data(), dev.d_atom_keep_mask,
               batch.total_atoms * sizeof(std::uint8_t), cudaMemcpyDeviceToHost);

    // Build submol that drops explicit H atoms
    return build_submol_from_keep_mask(mol, batch, /*molIndex=*/0);
}

// ---------------------------------------------------------------------
// gpu_kernel_clear_stereo
//   - For now, this is a CPU shim using RDKit's RemoveStereochemistry.
//   - When you later encode stereo into atom_flags and interpret it,
//     you can route this through clear_stereo_kernel too.
// ---------------------------------------------------------------------
ROMol gpu_kernel_clear_stereo(const ROMol &mol) {
    ROMol out(mol);
    RDKit::MolOps::removeStereochemistry(out);
    return out;
}

