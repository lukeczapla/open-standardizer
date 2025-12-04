// kernels.cu

#include "gpu_kernels.hpp"

#include <cuda_runtime.h>
#include <RDGeneral/Exceptions.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/RWMol.h>

#include <GraphMol/Atom.h>
#include <GraphMol/Bond.h>

#include "gpu_structs.hpp"
#include "tautomer_motif_tags.hpp"

#include <vector>
#include <algorithm>

// Device kernels from other .cu files:
extern "C" __global__
void stereo_kernel(
    const GAtom *atoms,
    const GBond *bonds,
    StereoFixResult *atomOut,
    StereoFixResult *bondOut,
    int numAtoms,
    int numBonds
);

extern "C" __global__
void tautomerize_kernel(
    const GAtom *atoms,
    const GBond *bonds,
    int8_t  *outCharges,
    uint8_t *outBondOrder,
    uint8_t *outAromatic,
    int      numAtoms,
    int      numBonds
);

// Convert RDKit molecule into GPU buffers (single-mol case)
static void mol_to_gpu_arrays(
    const RDKit::ROMol &mol,
    std::vector<GAtom> &atoms,
    std::vector<GBond> &bonds
) {
    const unsigned int nAtoms = mol.getNumAtoms();
    const unsigned int nBonds = mol.getNumBonds();

    atoms.clear();
    bonds.clear();
    atoms.reserve(nAtoms);
    bonds.reserve(nBonds);

    // 1) Run RDKit SMARTS-based tagging to get tautomer flags
    std::vector<uint16_t> atomFlags;
    std::vector<uint16_t> bondFlags;
    tagTautomerMotifs(mol, atomFlags, bondFlags);  // fills/assigns

    // Sanity: if tagTautomerMotifs didn't resize, ensure zero flags
    if (atomFlags.size() != nAtoms) atomFlags.assign(nAtoms, ATF_NONE);
    if (bondFlags.size() != nBonds) bondFlags.assign(nBonds, BTF_NONE);

    // atoms
    for (unsigned int i = 0; i < nAtoms; ++i) {
        const auto *a = mol.getAtomWithIdx(i);

        GAtom ga{};
        ga.atomicNum    = static_cast<uint8_t>(a->getAtomicNum());
        ga.formalCharge = static_cast<int8_t>(a->getFormalCharge());
        ga.degree       = static_cast<uint8_t>(a->getDegree());
        ga.valence      = static_cast<uint8_t>(a->getTotalValence());
        ga.hCount       = static_cast<uint8_t>(a->getTotalNumHs());

        // Map RDKit chiral tag to small code for stereo kernel
        switch (a->getChiralTag()) {
        case RDKit::Atom::CHI_TETRAHEDRAL_CW:
            ga.chiralFlag = 1;
            break;
        case RDKit::Atom::CHI_TETRAHEDRAL_CCW:
            ga.chiralFlag = 2;
            break;
        case RDKit::Atom::CHI_OTHER:
            ga.chiralFlag = 3; // “has stereo but not CW/CCW” if you care
            break;
        case RDKit::Atom::CHI_UNSPECIFIED:
        default:
            ga.chiralFlag = 0;
            break;
        }

        ga.tautomerFlags = atomFlags[i];

        atoms.push_back(ga);
    }

    // bonds
    for (unsigned int i = 0; i < nBonds; ++i) {
        const auto *b = mol.getBondWithIdx(i);

        GBond gb{};
        gb.idxA = static_cast<uint16_t>(b->getBeginAtomIdx());
        gb.idxB = static_cast<uint16_t>(b->getEndAtomIdx());

        // Bond order as small int (1,2,3). For aromatic, RDKit returns ~1.5;
        // you can either round or handle separately.
        double bt = b->getBondTypeAsDouble();
        if (bt <= 1.5)      gb.bondOrder = 1;
        else if (bt <= 2.5) gb.bondOrder = 2;
        else                gb.bondOrder = 3;

        gb.aromaticFlag = b->getIsAromatic() ? 1 : 0;

        // Stereo for your stereo kernel
        auto st = b->getStereo();
        switch (st) {
        case RDKit::Bond::STEREOE:
            gb.stereo = 3;  // your earlier code used 3 for E
            break;
        case RDKit::Bond::STEREOZ:
            gb.stereo = 4;  // and 4 for Z
            break;
        default:
            gb.stereo = 0;
            break;
        }

        gb.tautomerFlags = bondFlags[i];

        bonds.push_back(gb);
    }
}

// Apply GPU results → RDKit molecule
static void apply_stereo_back(
    RDKit::ROMol &mol,
    const std::vector<StereoFixResult> &atoms,
    const std::vector<StereoFixResult> &bonds
) {
    // atoms
    for (size_t i = 0; i < atoms.size(); i++) {
        const auto &r = atoms[i];
        auto a = mol.getAtomWithIdx(i);

        switch (r.atomChiral) {
        case 1: a->setChiralTag(RDKit::Atom::CHI_TETRAHEDRAL_CW);  break;
        case 2: a->setChiralTag(RDKit::Atom::CHI_TETRAHEDRAL_CCW); break;
        default: a->setChiralTag(RDKit::Atom::CHI_UNSPECIFIED);    break;
        }
    }

    // bonds
    for (size_t i = 0; i < bonds.size(); i++) {
        const auto &r = bonds[i];
        auto b = mol.getBondWithIdx(i);

        switch (r.bondStereo) {
        case 3: b->setStereo(RDKit::Bond::STEREOE); break;
        case 4: b->setStereo(RDKit::Bond::STEREOZ); break;
        default: b->setStereo(RDKit::Bond::STEREONONE); break;
        }
    }
}


// ============================================================
// PUBLIC ENTRYPOINT
// ============================================================
RDKit::ROMol gpu_kernel_stereo(const RDKit::ROMol &inMol) {
    RDKit::ROMol mol(inMol);  // copy so we can mutate

    // Host-side buffers
    std::vector<GAtom> hAtoms;
    std::vector<GBond> hBonds;
    mol_to_gpu_arrays(mol, hAtoms, hBonds);

    const int nA = static_cast<int>(hAtoms.size());
    const int nB = static_cast<int>(hBonds.size());

    if (nA == 0 && nB == 0) {
        return mol;
    }

    // Device buffers
    GAtom *dAtoms = nullptr;
    GBond *dBonds = nullptr;
    StereoFixResult *dAtomOut = nullptr;
    StereoFixResult *dBondOut = nullptr;

    cudaMalloc(&dAtoms,   nA * sizeof(GAtom));
    cudaMalloc(&dBonds,   nB * sizeof(GBond));
    cudaMalloc(&dAtomOut, nA * sizeof(StereoFixResult));
    cudaMalloc(&dBondOut, nB * sizeof(StereoFixResult));

    cudaMemcpy(dAtoms, hAtoms.data(), nA * sizeof(GAtom), cudaMemcpyHostToDevice);
    cudaMemcpy(dBonds, hBonds.data(), nB * sizeof(GBond), cudaMemcpyHostToDevice);

    // Launch kernel
    int block = 128;
    int grid  = (std::max(nA, nB) + block - 1) / block;
    if (grid < 1) grid = 1;

    stereo_kernel<<<grid, block>>>(
        dAtoms, dBonds, dAtomOut, dBondOut, nA, nB
    );
    cudaDeviceSynchronize();

    // Collect results back
    std::vector<StereoFixResult> outAtoms(nA);
    std::vector<StereoFixResult> outBonds(nB);

    cudaMemcpy(outAtoms.data(), dAtomOut, nA * sizeof(StereoFixResult), cudaMemcpyDeviceToHost);
    cudaMemcpy(outBonds.data(), dBondOut, nB * sizeof(StereoFixResult), cudaMemcpyDeviceToHost);

    cudaFree(dAtoms);
    cudaFree(dBonds);
    cudaFree(dAtomOut);
    cudaFree(dBondOut);

    apply_stereo_back(mol, outAtoms, outBonds);
    return mol;
}



RDKit::ROMol gpu_kernel_tautomerizer(const RDKit::ROMol &inMol) {
    RDKit::RWMol mol(inMol);  // RWMol, we will modify charges/bond orders

    std::vector<GAtom> hAtoms;
    std::vector<GBond> hBonds;
    mol_to_gpu_arrays(mol, hAtoms, hBonds);

    int nA = static_cast<int>(hAtoms.size());
    int nB = static_cast<int>(hBonds.size());

    // Nothing to do:
    if (nA == 0 && nB == 0) {
        return RDKit::ROMol(mol);
    }

    GAtom *dAtoms = nullptr;
    GBond *dBonds = nullptr;
    int8_t  *dCharges = nullptr;
    uint8_t *dOrder   = nullptr;
    uint8_t *dArom    = nullptr;

    cudaMalloc(&dAtoms,   nA * sizeof(GAtom));
    cudaMalloc(&dBonds,   nB * sizeof(GBond));
    cudaMalloc(&dCharges, nA * sizeof(int8_t));
    cudaMalloc(&dOrder,   nB * sizeof(uint8_t));
    cudaMalloc(&dArom,    nB * sizeof(uint8_t));

    cudaMemcpy(dAtoms, hAtoms.data(), nA * sizeof(GAtom), cudaMemcpyHostToDevice);
    cudaMemcpy(dBonds, hBonds.data(), nB * sizeof(GBond), cudaMemcpyHostToDevice);

    int block = 128;
    int grid  = (nA + nB + block - 1) / block;

    tautomerize_kernel<<<grid, block>>>(
        dAtoms, dBonds, dCharges, dOrder, dArom, nA, nB
    );
    cudaDeviceSynchronize();

    std::vector<int8_t>  outCharges(nA);
    std::vector<uint8_t> outOrder(nB), outArom(nB);

    cudaMemcpy(outCharges.data(), dCharges, nA * sizeof(int8_t),  cudaMemcpyDeviceToHost);
    cudaMemcpy(outOrder.data(),   dOrder,   nB * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(outArom.data(),    dArom,    nB * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cudaFree(dAtoms);
    cudaFree(dBonds);
    cudaFree(dCharges);
    cudaFree(dOrder);
    cudaFree(dArom);

    // Apply results back to mol: charges + bond order/aromatic
    for (unsigned int i = 0; i < mol.getNumAtoms(); ++i) {
        auto *a = mol.getAtomWithIdx(i);
        a->setFormalCharge(static_cast<int>(outCharges[i]));
    }

    for (unsigned int i = 0; i < mol.getNumBonds(); ++i) {
        auto *b = mol.getBondWithIdx(i);

        uint8_t ord = outOrder[i];
        if (ord == 1) b->setBondType(RDKit::Bond::SINGLE);
        else if (ord == 2) b->setBondType(RDKit::Bond::DOUBLE);
        else if (ord == 3) b->setBondType(RDKit::Bond::TRIPLE);

        bool arom = (outArom[i] != 0);
        b->setIsAromatic(arom);
    }

    return RDKit::ROMol(mol);
}
