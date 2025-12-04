from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.rdchem import BondStereo

from .enhanced_smiles import ChemAxonMeta, enhanced_smiles_to_mol

try:
    from rdkit.Chem import rdCIPLabeler
    HAS_CIP_LABELER = True
except ImportError:
    HAS_CIP_LABELER = False


@dataclass
class AtomCIPComparison:
    atom_index: int
    assignment: str       # ChemAxon code (e.g. "s", "r", "p")
    cip_code: Optional[str]  # RDKit code ("R"/"S") or None
    status: str           # "match", "mismatch", "missing_cip", "non_cip_code", ...


@dataclass
class BondCIPComparison:
    atom_index1: int      # ChemAxon A index 1
    atom_index2: int      # ChemAxon A index 2
    assignment: str       # ChemAxon code ("e", "z", ...)
    cip_code: Optional[str]  # RDKit code ("E"/"Z") or None
    status: str           # "match", "mismatch", "missing_cip", "no_bond", ...


@dataclass
class StereoValidationResult:
    atom_matches: List[AtomCIPComparison] = field(default_factory=list)
    atom_mismatches: List[AtomCIPComparison] = field(default_factory=list)
    atom_missing_cip: List[AtomCIPComparison] = field(default_factory=list)
    atom_non_cip_assignments: List[AtomCIPComparison] = field(default_factory=list)

    bond_matches: List[BondCIPComparison] = field(default_factory=list)
    bond_mismatches: List[BondCIPComparison] = field(default_factory=list)
    bond_missing_cip: List[BondCIPComparison] = field(default_factory=list)
    bond_unknown_assignments: List[BondCIPComparison] = field(default_factory=list)

    def ok(self) -> bool:
        return not self.atom_mismatches and not self.bond_mismatches


def _normalize_atom_expected(code: str) -> Optional[str]:
    c = (code or "").strip().upper()
    if c in ("R", "S"):
        return c
    return None  # e.g. "p" → non-CIP code


def _normalize_bond_expected(code: str) -> Optional[str]:
    c = (code or "").strip().upper()
    if c in ("E", "Z"):
        return c
    return None


def _to_rdkit_index(idx: int, index_base: int) -> int:
    # ChemAxon is 0-based in your data, so index_base=0 means "as-is"
    return idx if index_base == 0 else idx - 1


def _assign_cip(mol: Chem.Mol) -> None:
    # Always compute stereochemistry (atoms + bonds)
    rdmolops.AssignStereochemistry(
        mol, cleanIt=True, force=True, flagPossibleStereoCenters=True
    )
    # Then, if available, have CIPLabeler assign _CIPCode to atoms
    if HAS_CIP_LABELER:
        rdCIPLabeler.AssignCIPLabels(mol)


def _bond_stereo_to_ez(bond: Chem.Bond) -> Optional[str]:
    st = bond.GetStereo()
    if st == BondStereo.STEREOE:
        return "E"
    if st == BondStereo.STEREOZ:
        return "Z"
    return None


def validate_cip_assignments_on_mol(
    mol: Chem.Mol,
    meta: ChemAxonMeta,
    index_base: int = 0,
) -> StereoValidationResult:
    res = StereoValidationResult()
    if meta is None:
        return res

    _assign_cip(mol)
    n_atoms = mol.GetNumAtoms()

    # ---- atoms (A<idx>=code) ----
    for assign in meta.assignments:
        rd_idx = _to_rdkit_index(assign.atom_index, index_base=index_base)
        if rd_idx < 0 or rd_idx >= n_atoms:
            res.atom_non_cip_assignments.append(
                AtomCIPComparison(assign.atom_index, assign.config, None, "index_out_of_range")
            )
            continue

        expected = _normalize_atom_expected(assign.config)
        if expected is None:
            # e.g. "p": ChemAxon code we don't map to R/S
            res.atom_non_cip_assignments.append(
                AtomCIPComparison(assign.atom_index, assign.config, None, "non_cip_code")
            )
            continue

        atom = mol.GetAtomWithIdx(rd_idx)
        if atom.HasProp("_CIPCode"):
            cip = atom.GetProp("_CIPCode").upper()
        else:
            res.atom_missing_cip.append(
                AtomCIPComparison(assign.atom_index, assign.config, None, "missing_cip")
            )
            continue

        if cip == expected:
            res.atom_matches.append(
                AtomCIPComparison(assign.atom_index, assign.config, cip, "match")
            )
        else:
            res.atom_mismatches.append(
                AtomCIPComparison(assign.atom_index, assign.config, cip, "mismatch")
            )

    # ---- bonds (B<i,j>=code) ----
    for bassign in meta.bond_assignments:
        rd_i1 = _to_rdkit_index(bassign.atom_index1, index_base=index_base)
        rd_i2 = _to_rdkit_index(bassign.atom_index2, index_base=index_base)
        if not (0 <= rd_i1 < n_atoms and 0 <= rd_i2 < n_atoms):
            res.bond_unknown_assignments.append(
                BondCIPComparison(
                    bassign.atom_index1, bassign.atom_index2,
                    bassign.config, None, "index_out_of_range"
                )
            )
            continue

        expected = _normalize_bond_expected(bassign.config)
        if expected is None:
            res.bond_unknown_assignments.append(
                BondCIPComparison(
                    bassign.atom_index1, bassign.atom_index2,
                    bassign.config, None, "unknown_expected_code"
                )
            )
            continue

        bond = mol.GetBondBetweenAtoms(rd_i1, rd_i2)
        if bond is None:
            res.bond_unknown_assignments.append(
                BondCIPComparison(
                    bassign.atom_index1, bassign.atom_index2,
                    bassign.config, None, "no_bond"
                )
            )
            continue

        cip = _bond_stereo_to_ez(bond)
        if cip is None:
            res.bond_missing_cip.append(
                BondCIPComparison(
                    bassign.atom_index1, bassign.atom_index2,
                    bassign.config, None, "missing_cip"
                )
            )
            continue

        if cip == expected:
            res.bond_matches.append(
                BondCIPComparison(
                    bassign.atom_index1, bassign.atom_index2,
                    bassign.config, cip, "match"
                )
            )
        else:
            res.bond_mismatches.append(
                BondCIPComparison(
                    bassign.atom_index1, bassign.atom_index2,
                    bassign.config, cip, "mismatch"
                )
            )

    return res


def validate_cip_assignments_on_smiles(
    smiles: str,
    index_base: int = 0,
) -> Optional[StereoValidationResult]:
    mol, meta = enhanced_smiles_to_mol(smiles, index_base=index_base)
    if mol is None or meta is None:
        return None
    return validate_cip_assignments_on_mol(mol, meta, index_base=index_base)
