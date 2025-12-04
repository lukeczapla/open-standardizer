// src/gpu/tautomer_motif_tags.cpp

#include "tautomer_motif_tags.hpp"

#include <GraphMol/ROMol.h>
#include <GraphMol/SmartsParse/SmartsParse.h>
#include <GraphMol/Substruct/SubstructMatch.h>

#include "gpu_structs.hpp"  // AtomTautomerFlags, BondTautomerFlags

using namespace RDKit;

// simple holder
struct MotifDef {
    std::string name;
    std::string smarts;
};

static const std::vector<MotifDef> &getMotifDefs() {
    // These are deliberately “coarse” patterns aligned with
    // RDKit’s tautomer scoring substructures (C=O, N=O, P=O,
    // C=hetero, C(=hetero)-hetero, aci-nitro, etc.).
    // You can expand / tweak them if needed.

    static const std::vector<MotifDef> defs = {
        {"C=O",               "[#6]=[#8]"},
        {"C=S",               "[#6]=[#16]"},
        {"N=O",               "[#7]=[#8]"},
        {"P=O",               "[#15]=[#8]"},
        {"C=hetero",          "[#6]=[!#1;!#6]"},
        {"C(=hetero)-hetero", "[#6](=[!#1;!#6])[!#1;!#6]"},
        {"aci-nitro",         "[#6]=[N+]([O-])[OH]"},
        {"nitro_generic",     "[N+](=O)[O-]"}
    };
    return defs;
}

void tagTautomerMotifs(
    const ROMol &mol,
    std::vector<uint16_t> &atomFlags,
    std::vector<uint16_t> &bondFlags)
{
    const auto nAtoms = mol.getNumAtoms();
    const auto nBonds = mol.getNumBonds();

    atomFlags.assign(nAtoms, ATF_NONE);
    bondFlags.assign(nBonds, BTF_NONE);

    SubstructMatchParameters params;
    params.useChirality = false;
    params.maxMatches = 0;  // unlimited

    for (const auto &def : getMotifDefs()) {
        std::unique_ptr<ROMol> query(SmartsToMol(def.smarts));
        if (!query) continue;

        const auto matches = SubstructMatch(mol, *query, params);
        if (matches.empty()) continue;

        for (const auto &match : matches) {
            // match : vector<pair<int queryIdx, int molIdx>>
            // We’ll tag atoms and any bonds between matched atoms.
            if (def.name == "C=O" || def.name == "C=S") {
                // Both atoms on this bond are part of a keto/thioketo system
                for (size_t i = 0; i + 1 < match.size(); ++i) {
                    auto mi1 = match[i].second;
                    auto mi2 = match[i + 1].second;
                    auto *a1 = mol.getAtomWithIdx(mi1);
                    auto *a2 = mol.getAtomWithIdx(mi2);

                    if (a1->getAtomicNum() == 6) {
                        atomFlags[mi1] |= ATF_TAUT_KETO_CENTER;
                    }
                    if (a2->getAtomicNum() == 6) {
                        atomFlags[mi2] |= ATF_TAUT_KETO_CENTER;
                    }
                    if (a1->getAtomicNum() == 8 || a1->getAtomicNum() == 16) {
                        atomFlags[mi1] |= ATF_TAUT_ENOL_O;
                    }
                    if (a2->getAtomicNum() == 8 || a2->getAtomicNum() == 16) {
                        atomFlags[mi2] |= ATF_TAUT_ENOL_O;
                    }

                    const Bond *b = mol.getBondBetweenAtoms(mi1, mi2);
                    if (b) {
                        bondFlags[b->getIdx()] |= BTF_TAUT_KETO_ENOL_PATH | BTF_TAUT_C_EQ_HETERO;
                    }
                }
            } else if (def.name == "N=O") {
                for (size_t i = 0; i + 1 < match.size(); ++i) {
                    auto mi1 = match[i].second;
                    auto mi2 = match[i + 1].second;
                    auto *a1 = mol.getAtomWithIdx(mi1);
                    auto *a2 = mol.getAtomWithIdx(mi2);

                    if (a1->getAtomicNum() == 7) atomFlags[mi1] |= ATF_TAUT_N_OXIDE;
                    if (a2->getAtomicNum() == 7) atomFlags[mi2] |= ATF_TAUT_N_OXIDE;

                    const Bond *b = mol.getBondBetweenAtoms(mi1, mi2);
                    if (b) {
                        bondFlags[b->getIdx()] |= BTF_TAUT_N_OXIDE_PATH;
                    }
                }
            } else if (def.name == "P=O") {
                for (size_t i = 0; i + 1 < match.size(); ++i) {
                    auto mi1 = match[i].second;
                    auto mi2 = match[i + 1].second;
                    auto *a1 = mol.getAtomWithIdx(mi1);
                    auto *a2 = mol.getAtomWithIdx(mi2);

                    if (a1->getAtomicNum() == 15) atomFlags[mi1] |= ATF_TAUT_P_CENTER;
                    if (a2->getAtomicNum() == 15) atomFlags[mi2] |= ATF_TAUT_P_CENTER;

                    const Bond *b = mol.getBondBetweenAtoms(mi1, mi2);
                    if (b) {
                        bondFlags[b->getIdx()] |= BTF_TAUT_P_O_PATH;
                    }
                }
            } else if (def.name == "C=hetero") {
                for (size_t i = 0; i + 1 < match.size(); ++i) {
                    auto mi1 = match[i].second;
                    auto mi2 = match[i + 1].second;
                    const Bond *b = mol.getBondBetweenAtoms(mi1, mi2);
                    if (b) {
                        bondFlags[b->getIdx()] |= BTF_TAUT_C_EQ_HETERO;
                    }
                }
            } else if (def.name == "C(=hetero)-hetero") {
                // multi-atom pattern; mark all bonds between matched atoms
                for (size_t i = 0; i < match.size(); ++i) {
                    for (size_t j = i + 1; j < match.size(); ++j) {
                        auto mi1 = match[i].second;
                        auto mi2 = match[j].second;
                        const Bond *b = mol.getBondBetweenAtoms(mi1, mi2);
                        if (b) {
                            bondFlags[b->getIdx()] |= BTF_TAUT_CEQHETERO_HET;
                        }
                    }
                }
            } else if (def.name == "aci-nitro" || def.name == "nitro_generic") {
                // Tag nitration motifs: one N, two Os, maybe carbon
                std::vector<unsigned int> atomIdxs;
                atomIdxs.reserve(match.size());
                for (auto &p : match) atomIdxs.push_back(p.second);

                for (auto idx : atomIdxs) {
                    const auto *a = mol.getAtomWithIdx(idx);
                    if (a->getAtomicNum() == 7) {
                        atomFlags[idx] |= ATF_TAUT_NITRO_N;
                    } else if (a->getAtomicNum() == 8) {
                        atomFlags[idx] |= ATF_TAUT_NITRO_O;
                    }
                }

                // Mark all bonds connecting these atoms as nitro path
                for (size_t i = 0; i < atomIdxs.size(); ++i) {
                    for (size_t j = i + 1; j < atomIdxs.size(); ++j) {
                        const Bond *b = mol.getBondBetweenAtoms(atomIdxs[i], atomIdxs[j]);
                        if (b) {
                            bondFlags[b->getIdx()] |= BTF_TAUT_NITRO_PATH;
                        }
                    }
                }
            }
        }
    }

    // Optional: tag generic “hetero with mobile H” for scoring-like effects
    for (auto atom : mol.atoms()) {
        auto idx = atom->getIdx();
        auto anum = atom->getAtomicNum();
        if (anum == 7 || anum == 8 || anum == 16 || anum == 34 || anum == 52) {
            if (atom->getTotalNumHs() > 0) {
                atomFlags[idx] |= ATF_TAUT_HETERO_HOT;
            }
        }
    }
}
