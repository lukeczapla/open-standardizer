from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import re

from rdkit import Chem
from rdkit.Chem.rdchem import StereoGroup, StereoGroupType

# ---------- data structures ----------

@dataclass
class StereoAssignment:
    atom_index: int   # e.g. 7 in A7=s
    config: str       # e.g. "s", "r", "p"

@dataclass
class BondAssignment:
    atom_index1: int  # e.g. 0 in B0,1=e
    atom_index2: int  # e.g. 1 in B0,1=e
    config: str       # e.g. "e", "z"

@dataclass
class StereoGroupSpec:
    group_type: str           # "a", "o", "&"
    group_index: Optional[int]
    atom_indices: List[int]

@dataclass
class ChemAxonMeta:
    raw: str
    tokens: List[str] = field(default_factory=list)
    assignments: List[StereoAssignment] = field(default_factory=list)
    bond_assignments: List[BondAssignment] = field(default_factory=list)
    stereo_groups: List[StereoGroupSpec] = field(default_factory=list)
    other_tokens: List[str] = field(default_factory=list)

    def to_raw(self) -> str:
        return self.raw

    def __bool__(self) -> bool:
        return bool(self.raw.strip())

@dataclass
class ChemAxonEnhanced:
    core_smiles: str
    meta: Optional[ChemAxonMeta] = None

    def to_string(self, preserve_meta: bool = True) -> str:
        core = self.core_smiles.strip()
        if preserve_meta and self.meta and self.meta.raw.strip():
            return f"{core} {{{self.meta.to_raw()}}}"
        return core

# ---------- parsing ----------

_ASSIGN_RE      = re.compile(r"^A(\d+)=([A-Za-z0-9_]+)$")
_BOND_ASSIGN_RE = re.compile(r"^B(\d+),(\d+)=([A-Za-z0-9_]+)$")
_GROUP_RE       = re.compile(r"^([ao&])(\d*):([\d,]+)$")

def _parse_meta_block(block: str) -> ChemAxonMeta:
    raw = block.strip()
    if not raw:
        return ChemAxonMeta(raw="")

    tokens = [t.strip() for t in raw.split(";") if t.strip()]
    assignments: List[StereoAssignment] = []
    bond_assignments: List[BondAssignment] = []
    groups: List[StereoGroupSpec] = []
    others: List[str] = []

    for tok in tokens:
        m = _ASSIGN_RE.match(tok)
        if m:
            assignments.append(
                StereoAssignment(atom_index=int(m.group(1)), config=m.group(2))
            )
            continue

        m = _BOND_ASSIGN_RE.match(tok)
        if m:
            bond_assignments.append(
                BondAssignment(
                    atom_index1=int(m.group(1)),
                    atom_index2=int(m.group(2)),
                    config=m.group(3),
                )
            )
            continue

        m = _GROUP_RE.match(tok)
        if m:
            group_type = m.group(1)
            group_index = int(m.group(2)) if m.group(2) else None
            atom_indices = [int(x) for x in m.group(3).split(",") if x]
            groups.append(
                StereoGroupSpec(
                    group_type=group_type,
                    group_index=group_index,
                    atom_indices=atom_indices,
                )
            )
            continue

        others.append(tok)

    return ChemAxonMeta(
        raw=raw,
        tokens=tokens,
        assignments=assignments,
        bond_assignments=bond_assignments,
        stereo_groups=groups,
        other_tokens=others,
    )

def parse_chemaxon_enhanced(smiles: str) -> ChemAxonEnhanced:
    if smiles is None:
        return ChemAxonEnhanced(core_smiles="", meta=None)

    s = smiles.strip()
    if not s.endswith("}"):
        return ChemAxonEnhanced(core_smiles=s, meta=None)

    close_idx = len(s) - 1
    open_idx = s.rfind("{", 0, close_idx)
    if open_idx == -1:
        return ChemAxonEnhanced(core_smiles=s, meta=None)

    core = s[:open_idx].rstrip()
    block = s[open_idx + 1 : close_idx]
    meta = _parse_meta_block(block)
    return ChemAxonEnhanced(core_smiles=core, meta=meta)

def strip_to_core(smiles: str) -> str:
    return parse_chemaxon_enhanced(smiles).core_smiles

# ---------- RDKit mapping ----------

def _resolve_atom_indices(
    mol: Chem.Mol,
    indices: List[int],
    index_base: int = 0,
) -> List[Chem.Atom]:
    n = mol.GetNumAtoms()
    atoms: List[Chem.Atom] = []
    for i in indices:
        if index_base == 1:
            i = i - 1
        if 0 <= i < n:
            atoms.append(mol.GetAtomWithIdx(i))
    return atoms

def apply_chemaxon_meta_to_mol(
    mol: Chem.Mol,
    meta: ChemAxonMeta,
    index_base: int = 0,
    store_atom_assignments_prop: str = "_ChemAxonAtomStereo",
    store_bond_assignments_prop: str = "_ChemAxonBondStereo",
) -> Chem.Mol:
    if meta is None or not meta.raw.strip():
        return mol

    # A<idx>=code → atom property
    for assign in meta.assignments:
        atoms = _resolve_atom_indices(mol, [assign.atom_index], index_base=index_base)
        if not atoms:
            continue
        atoms[0].SetProp(store_atom_assignments_prop, assign.config)

    # B<i,j>=code → bond property
    n_atoms = mol.GetNumAtoms()
    for bassign in meta.bond_assignments:
        i1 = bassign.atom_index1 + (0 if index_base == 0 else -1)
        i2 = bassign.atom_index2 + (0 if index_base == 0 else -1)
        if not (0 <= i1 < n_atoms and 0 <= i2 < n_atoms):
            continue
        bond = mol.GetBondBetweenAtoms(i1, i2)
        if bond is not None:
            bond.SetProp(store_bond_assignments_prop, bassign.config)

    # groups: a/o/&
    rdkit_groups: List[StereoGroup] = []
    for grp in meta.stereo_groups:
        if grp.group_type == "a":
            gtype = StereoGroupType.STEREO_ABSOLUTE
        elif grp.group_type == "o":
            gtype = StereoGroupType.STEREO_OR
        elif grp.group_type == "&":
            gtype = StereoGroupType.STEREO_AND
        else:
            continue
        atoms = _resolve_atom_indices(mol, grp.atom_indices, index_base=index_base)
        if not atoms:
            continue
        rdkit_groups.append(StereoGroup(gtype, atoms))

    if rdkit_groups:
        mol.SetStereoGroups(rdkit_groups)

    mol.SetProp("_ChemAxonMetaRaw", meta.raw)
    return mol

def enhanced_smiles_to_mol(
    smiles: str,
    index_base: int = 0,
) -> Tuple[Optional[Chem.Mol], Optional[ChemAxonMeta]]:
    parsed = parse_chemaxon_enhanced(smiles)
    if not parsed.core_smiles:
        return None, None

    mol = Chem.MolFromSmiles(parsed.core_smiles)
    if mol is None:
        return None, None

    if parsed.meta:
        mol = apply_chemaxon_meta_to_mol(mol, parsed.meta, index_base=index_base)

    return mol, parsed.meta
