# standardizing_report.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.rdchem import BondStereo

from .enhanced_smiles import enhanced_smiles_to_mol, ChemAxonMeta
from .stereo_export import export_enhanced_smiles_from_mol
from .standardize import standardize, DEFAULT_OPS
from .gpu_cpu_policy_manager import GPU_CPU_MANAGER, Policy


@dataclass
class AtomState:
    idx: int
    atomic_num: int
    formal_charge: int
    isotope: int
    aromatic: bool
    cip: Optional[str]


@dataclass
class BondState:
    idx: int
    begin_atom: int
    end_atom: int
    order: float
    aromatic: bool
    stereo: Optional[str]   # "E"/"Z"/None


@dataclass
class AtomChange:
    idx: int
    before: AtomState
    after: AtomState


@dataclass
class BondChange:
    idx: int
    before: BondState
    after: BondState


@dataclass
class StandardizationReport:
    input_enhanced: str
    output_enhanced: str
    input_core_smiles: str
    output_core_smiles: str

    atom_changes: List[AtomChange] = field(default_factory=list)
    bond_changes: List[BondChange] = field(default_factory=list)

    meta_in: Optional[ChemAxonMeta] = None
    meta_out_raw_block: Optional[str] = None  # the {...} contents for output


try:
    from rdkit.Chem import rdCIPLabeler
    HAS_CIP_LABELER = True
except ImportError:
    HAS_CIP_LABELER = False


def _assign_stereo(mol: Chem.Mol) -> None:
    rdmolops.AssignStereochemistry(
        mol, cleanIt=True, force=True, flagPossibleStereoCenters=True
    )
    if HAS_CIP_LABELER:
        rdCIPLabeler.AssignCIPLabels(mol)

def _bond_stereo_to_ez(bond: Chem.Bond) -> Optional[str]:
    st = bond.GetStereo()
    if st == BondStereo.STEREOE:
        return "E"
    if st == BondStereo.STEREOZ:
        return "Z"
    return None


def snapshot_atoms(mol: Chem.Mol) -> List[AtomState]:
    _assign_stereo(mol)
    atoms: List[AtomState] = []
    for a in mol.GetAtoms():
        idx = a.GetIdx()
        cip = a.GetProp("_CIPCode").upper() if a.HasProp("_CIPCode") else None
        atoms.append(
            AtomState(
                idx=idx,
                atomic_num=a.GetAtomicNum(),
                formal_charge=a.GetFormalCharge(),
                isotope=a.GetIsotope(),
                aromatic=a.GetIsAromatic(),
                cip=cip,
            )
        )
    return atoms


def snapshot_bonds(mol: Chem.Mol) -> List[BondState]:
    _assign_stereo(mol)
    bonds: List[BondState] = []
    for b in mol.GetBonds():
        idx = b.GetIdx()
        bonds.append(
            BondState(
                idx=idx,
                begin_atom=b.GetBeginAtomIdx(),
                end_atom=b.GetEndAtomIdx(),
                order=float(b.GetBondTypeAsDouble()),
                aromatic=b.GetIsAromatic(),
                stereo=_bond_stereo_to_ez(b),
            )
        )
    return bonds

def diff_atoms(before: List[AtomState], after: List[AtomState]) -> List[AtomChange]:
    # Only compare up to the min length; if you strip fragments,
    # some atoms just "disappear" from the tail.
    n = min(len(before), len(after))
    changes: List[AtomChange] = []
    for i in range(n):
        b = before[i]
        a = after[i]
        if (
            b.atomic_num != a.atomic_num
            or b.formal_charge != a.formal_charge
            or b.isotope != a.isotope
            or b.aromatic != a.aromatic
            or (b.cip or "").upper() != (a.cip or "").upper()
        ):
            changes.append(AtomChange(idx=i, before=b, after=a))
    return changes


def diff_bonds(before: List[BondState], after: List[BondState]) -> List[BondChange]:
    m = min(len(before), len(after))
    changes: List[BondChange] = []
    for i in range(m):
        b = before[i]
        a = after[i]
        if (
            b.begin_atom != a.begin_atom
            or b.end_atom != a.end_atom
            or abs(b.order - a.order) > 1e-3
            or b.aromatic != a.aromatic
            or (b.stereo or "") != (a.stereo or "")
        ):
            changes.append(BondChange(idx=i, before=b, after=a))
    return changes

def standardize_enhanced_with_diff(
    enhanced_smiles: str,
    ops: Optional[List[str]] = None,
    policy: Policy = GPU_CPU_MANAGER,
    index_base: int = 0,
    export_mode: str = "replace",
) -> Optional[StandardizationReport]:
    """
    End-to-end:

      ChemAxon-enhanced SMILES
        -> parse to mol + meta
        -> run open-standardizer standardize() on the core SMILES
        -> rebuild mol_out
        -> diff core structure
        -> export *new* enhanced SMILES with A/B/& codes from RDKit

    - ops: list of operation names (defaults to DEFAULT_OPS)
    - policy: your GPU/CPU Policy
    - index_base: 0 for ChemAxon-style indices (B0,1=e), 1 if you ever
                  want to round-trip with 1-based indices.
    - export_mode: "append" or "replace" for merging with existing meta.

    Returns a StandardizationReport, or None if we fail to parse.
    """
    ops = ops or DEFAULT_OPS

    # 1) Parse enhanced SMILES
    mol_in, meta_in = enhanced_smiles_to_mol(enhanced_smiles, index_base=index_base)
    if mol_in is None:
        return None

    core_in = Chem.MolToSmiles(mol_in, canonical=True)

    # 2) Standardize via existing pipeline (GPU+CPU)
    core_out = standardize(core_in, ops=ops, policy=policy)
    if core_out is None:
        return None

    mol_out = Chem.MolFromSmiles(core_out)
    if mol_out is None:
        return None

    # 3) Snapshot before/after
    atoms_before = snapshot_atoms(mol_in)
    bonds_before = snapshot_bonds(mol_in)
    atoms_after = snapshot_atoms(mol_out)
    bonds_after = snapshot_bonds(mol_out)

    atom_changes = diff_atoms(atoms_before, atoms_after)
    bond_changes = diff_bonds(bonds_before, bonds_after)

    # 4) Export enhanced SMILES for mol_out with a standard "{...}" block
    enhanced_out = export_enhanced_smiles_from_mol(
        mol_out,
        existing_meta=meta_in,
        index_base=index_base,
        mode=export_mode,  # "replace" to refresh A/B/&; "append" to keep unknown tokens
    )

    # We can also expose just the raw curly block if you want to compare with ChemAxon.
    meta_out_block = enhanced_out.split("{", 1)[1].rsplit("}", 1)[0] if "{" in enhanced_out else ""

    return StandardizationReport(
        input_enhanced=enhanced_smiles,
        output_enhanced=enhanced_out,
        input_core_smiles=core_in,
        output_core_smiles=core_out,
        atom_changes=atom_changes,
        bond_changes=bond_changes,
        meta_in=meta_in,
        meta_out_raw_block=meta_out_block or None,
    )
