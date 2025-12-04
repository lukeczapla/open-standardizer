// src/gpu/core_ops_batch.cpp

#include "core_ops_batch.hpp"
#include "core_ops_flags.hpp"

#include <GraphMol/ROMol.h>
#include <GraphMol/Atom.h>
#include <GraphMol/Bond.h>

#include <vector>

using RDKit::ROMol;

namespace {

// Simple connected-components labelling for a single ROMol.
// Fills fragMapping[i] with the fragment index of atom i.
void compute_frag_ids(const ROMol &mol, std::vector<int> &fragMapping) {
    const unsigned int nAtoms = mol.getNumAtoms();
    fragMapping.assign(nAtoms, -1);

    // Build adjacency from bonds using the RDKit range-for API.
    std::vector<std::vector<unsigned int>> adj(nAtoms);
    for (auto bond : mol.bonds()) {
        const RDKit::Bond *b = bond;
        unsigned int a1 = b->getBeginAtomIdx();
        unsigned int a2 = b->getEndAtomIdx();
        adj[a1].push_back(a2);
        adj[a2].push_back(a1);
    }

    int currentFrag = 0;
    std::vector<unsigned int> stack;
    stack.reserve(nAtoms);

    for (unsigned int start = 0; start < nAtoms; ++start) {
        if (fragMapping[start] != -1) {
            continue;  // already visited
        }

        // Start a new fragment at 'start'
        stack.clear();
        stack.push_back(start);
        fragMapping[start] = currentFrag;

        while (!stack.empty()) {
            unsigned int idx = stack.back();
            stack.pop_back();

            for (unsigned int nbr : adj[idx]) {
                if (fragMapping[nbr] == -1) {
                    fragMapping[nbr] = currentFrag;
                    stack.push_back(nbr);
                }
            }
        }

        ++currentFrag;
    }
}

} // end anonymous namespace


CoreOpsBatch build_core_ops_batch_from_mols(
    const std::vector<const ROMol*>& mols
) {
    CoreOpsBatch batch;
    batch.num_mols = static_cast<int>(mols.size());

    batch.mol_atom_offset.resize(batch.num_mols);
    batch.mol_num_atoms.resize(batch.num_mols);

    // -----------------------------
    // 1) Compute per-mol offsets and total_atoms
    // -----------------------------
    int offset = 0;
    for (int m = 0; m < batch.num_mols; ++m) {
        const ROMol* mol = mols[m];
        if (!mol) {
            batch.mol_atom_offset[m] = offset;
            batch.mol_num_atoms[m]   = 0;
            continue;
        }
        int nAtoms = static_cast<int>(mol->getNumAtoms());
        batch.mol_atom_offset[m] = offset;
        batch.mol_num_atoms[m]   = nAtoms;
        offset += nAtoms;
    }
    batch.total_atoms = offset;

    batch.atom_isotope.resize(batch.total_atoms);
    batch.atom_flags.resize(batch.total_atoms);
    batch.atom_frag_id.resize(batch.total_atoms);
    batch.atom_keep_mask.resize(batch.total_atoms, 1u); // default: keep everything

    int global_idx = 0;
    int max_frags = 0;

    // -----------------------------
    // 2) Fill per-atom fields and fragment IDs
    // -----------------------------
    for (int m = 0; m < batch.num_mols; ++m) {
        const ROMol* mol = mols[m];
        int nAtoms = batch.mol_num_atoms[m];
        if (!mol || nAtoms == 0) {
            continue;
        }

        // Isotopes & flags
        for (int i = 0; i < nAtoms; ++i) {
            int idx = global_idx + i;
            const auto* atom = mol->getAtomWithIdx(i);
            batch.atom_isotope[idx] = atom->getIsotope();
            batch.atom_flags[idx]   = 0;  // TODO: encode stereo / explicit H if desired
        }

        // Fragment IDs via our own CC labelling
        std::vector<int> fragMapping;
        compute_frag_ids(*mol, fragMapping);

        int local_max_frag = 0;
        for (int i = 0; i < nAtoms; ++i) {
            int idx  = global_idx + i;
            int frag = fragMapping[i];
            if (frag < 0) frag = 0; // just in case
            batch.atom_frag_id[idx] = frag;
            if (frag > local_max_frag) {
                local_max_frag = frag;
            }
        }

        int numFrags = local_max_frag + 1;
        if (numFrags > max_frags) {
            max_frags = numFrags;
        }

        global_idx += nAtoms;
    }

    batch.max_frags_per_mol = (max_frags > 0 ? max_frags : 1);
    return batch;
}
