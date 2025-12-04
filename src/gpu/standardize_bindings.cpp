#define PY_SSIZE_T_CLEAN
#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>
#define PY_SSIZE_T_CLEAN

#include <RDGeneral/Exceptions.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>

#include "gpu_kernels.hpp"

namespace py = pybind11;

// Helper: Convert Python SMILES → RDKit Mol
static std::unique_ptr<RDKit::ROMol> mol_from_smiles(const std::string &s) {
    RDKit::ROMol *mol = RDKit::SmilesToMol(s);
    if (!mol) throw std::runtime_error("Invalid SMILES: " + s);
    return std::unique_ptr<RDKit::ROMol>(mol);
}

// Helper: RDKit Mol → SMILES
static std::string mol_to_smiles(const RDKit::ROMol &mol) {
    return RDKit::MolToSmiles(mol, true);
}

// Wraps actual CUDA kernel calls
static RDKit::ROMol gpu_apply_kernel(
        const RDKit::ROMol &mol,
        const std::function<RDKit::ROMol(const RDKit::ROMol &)> &fn,
        const std::string &name) {

    try {
        return fn(mol);
    } catch (const std::exception &e) {
        std::string msg = "[GPU] Kernel '" + name + "' failed: ";
        msg += e.what();
        throw std::runtime_error(msg);
    }
}

PYBIND11_MODULE(standardize_gpu, m) {
    m.doc() = "GPU accelerated RDKit standardizer kernels (pybind11 interface)";

    // --- Stereo ---
    m.def("gpu_stereo", [](const std::string &smiles) {
        auto mol = mol_from_smiles(smiles);
        RDKit::ROMol out = gpu_apply_kernel(
            *mol,
            gpu_kernel_stereo,
            "stereo"
        );
        return mol_to_smiles(out);
    });

    // --- Charge Normalization ---
    m.def("gpu_charge", [](const std::string &smiles) {
        auto mol = mol_from_smiles(smiles);
        RDKit::ROMol out = gpu_apply_kernel(
            *mol,
            gpu_kernel_charge,
            "charge"
        );
        return mol_to_smiles(out);
    });

    // --- Bond Order Inference ---
    m.def("gpu_bond_order", [](const std::string &smiles) {
        auto mol = mol_from_smiles(smiles);
        RDKit::ROMol out = gpu_apply_kernel(
            *mol,
            gpu_kernel_bond_order,
            "bond_order"
        );
        return mol_to_smiles(out);
    });

    // --- Aromaticity ---
    m.def("gpu_aromaticity", [](const std::string &smiles) {
        auto mol = mol_from_smiles(smiles);
        RDKit::ROMol out = gpu_apply_kernel(
            *mol,
            gpu_kernel_aromaticity,
            "aromaticity"
        );
        return mol_to_smiles(out);
    });

    // --- Mesomerizer / Resonance Canonicalization ---
    m.def("gpu_mesomerizer", [](const std::string &smiles) {
        auto mol = mol_from_smiles(smiles);
        RDKit::ROMol out = gpu_apply_kernel(
            *mol,
            gpu_kernel_mesomerizer,
            "mesomerizer"
        );
        return mol_to_smiles(out);
    });

    // --- Final clean-up ---
    m.def("gpu_final_clean", [](const std::string &smiles) {
        auto mol = mol_from_smiles(smiles);
        RDKit::ROMol out = gpu_apply_kernel(
            *mol,
            gpu_kernel_final_clean,
            "final_clean"
        );
        return mol_to_smiles(out);
    });

    m.def("gpu_clear_isotopes", [](const std::string &smiles) {
        auto mol = mol_from_smiles(smiles);
        RDKit::ROMol out = gpu_apply_kernel(*mol, gpu_kernel_clear_isotopes, "clear_isotopes");
        return mol_to_smiles(out);
    });

    m.def("gpu_clear_stereo_core", [](const std::string &smiles) {
        auto mol = mol_from_smiles(smiles);
        RDKit::ROMol out = gpu_apply_kernel(*mol, gpu_kernel_clear_stereo, "clear_stereo");
        return mol_to_smiles(out);
    });

    m.def("gpu_keep_largest_fragment", [](const std::string &smiles) {
        auto mol = mol_from_smiles(smiles);
        RDKit::ROMol out = gpu_apply_kernel(*mol, gpu_kernel_keep_largest_fragment,
                                            "remove_largest_fragment");
        return mol_to_smiles(out);
    });

    m.def("gpu_remove_explicit_h", [](const std::string &smiles) {
        auto mol = mol_from_smiles(smiles);
        RDKit::ROMol out = gpu_apply_kernel(*mol, gpu_kernel_remove_explicit_h,
                                            "remove_explicit_h");
        return mol_to_smiles(out);
    });

    m.def("gpu_tautomerize", [](const std::string &smiles) {
        auto mol = mol_from_smiles(smiles);
        RDKit::ROMol out = gpu_apply_kernel(
            *mol,
            gpu_kernel_tautomerizer,
            "tautomerizer"
        );
        return mol_to_smiles(out);
    });
}
