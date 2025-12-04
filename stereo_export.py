from __future__ import annotations

from typing import List, Optional

from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.rdchem import StereoGroupType, BondStereo

from .enhanced_smiles import ChemAxonMeta


# -------------------------------------------------------------
# Internal: consistent stereo assignment (atoms + bonds)
# -------------------------------------------------------------

try:
    from rdkit.Chem import rdCIPLabeler
    HAS_CIP_LABELER = True
except ImportError:
    HAS_CIP_LABELER = False


def _assign_stereo(mol: Chem.Mol) -> None:
    """
    Ensure RDKit has both atom and bond stereochemistry assigned.

    - AssignStereochemistry sets up double-bond E/Z and chiral flags
    - rdCIPLabeler (if present) assigns _CIPCode on atoms (R/S)
    """
    rdmolops.AssignStereochemistry(
        mol, cleanIt=True, force=True, flagPossibleStereoCenters=True
    )
    if HAS_CIP_LABELER:
        rdCIPLabeler.AssignCIPLabels(mol)


# -------------------------------------------------------------
# Atom CIP: A<i>=r/s
# -------------------------------------------------------------

def export_cip_assignments(mol: Chem.Mol, index_base: int = 0) -> List[str]:
    """
    Export atom CIP assignments as ChemAxon A<i>=r/s tokens.

    - index_base = 0  → RDKit/your ChemAxon data (0-based indices, A0=...)
    - index_base = 1  → 1-based indices, A1=..., A2=..., etc.
    """
    _assign_stereo(mol)

    tokens: List[str] = []
    for idx, atom in enumerate(mol.GetAtoms()):
        if not atom.HasProp("_CIPCode"):
            continue
        cip = atom.GetProp("_CIPCode").upper()
        if cip not in ("R", "S"):
            continue
        ax_idx = idx + 1 if index_base == 1 else idx
        tokens.append(f"A{ax_idx}={cip.lower()}")
    return tokens


# -------------------------------------------------------------
# Bond E/Z: B<i,j>=e/z
# -------------------------------------------------------------

def export_ez_assignments(mol: Chem.Mol, index_base: int = 0) -> List[str]:
    """
    Export double-bond E/Z assignments as ChemAxon B<i,j>=e/z tokens.

    Uses RDKit BondStereo:
        STEREOE -> e
        STEREOZ -> z
    """
    # _assign_stereo already called in export_cip_assignments, but it’s cheap.
    _assign_stereo(mol)

    tokens: List[str] = []
    for bond in mol.GetBonds():
        st = bond.GetStereo()
        if st == BondStereo.STEREOE:
            code = "e"
        elif st == BondStereo.STEREOZ:
            code = "z"
        else:
            continue

        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        if index_base == 1:
            i += 1
            j += 1
        tokens.append(f"B{i},{j}={code}")
    return tokens


# -------------------------------------------------------------
# Stereo groups: a/o/& tokens (your existing logic)
# -------------------------------------------------------------

def export_stereogroup_tokens(mol: Chem.Mol, index_base: int = 0) -> List[str]:
    groups = mol.GetStereoGroups()
    if not groups:
        return []

    abs_groups = []
    or_groups = []
    and_groups = []
    for g in groups:
        if g.GetGroupType() == StereoGroupType.STEREO_ABSOLUTE:
            abs_groups.append(g)
        elif g.GetGroupType() == StereoGroupType.STEREO_OR:
            or_groups.append(g)
        elif g.GetGroupType() == StereoGroupType.STEREO_AND:
            and_groups.append(g)

    tokens: List[str] = []

    # a:1,2,3
    for g in abs_groups:
        idxs = [a.GetIdx() for a in g.GetAtoms()]
        if index_base == 1:
            idxs = [i + 1 for i in idxs]
        tokens.append("a:" + ",".join(map(str, idxs)))

    # o1:... ; o2:...
    for gi, g in enumerate(or_groups, start=1):
        idxs = [a.GetIdx() for a in g.GetAtoms()]
        if index_base == 1:
            idxs = [i + 1 for i in idxs]
        tokens.append(f"o{gi}:" + ",".join(map(str, idxs)))

    # &1:... ; &2:...
    for gi, g in enumerate(and_groups, start=1):
        idxs = [a.GetIdx() for a in g.GetAtoms()]
        if index_base == 1:
            idxs = [i + 1 for i in idxs]
        tokens.append(f"&{gi}:" + ",".join(map(str, idxs)))

    return tokens


# -------------------------------------------------------------
# Merging into an existing { ... } block
# -------------------------------------------------------------

def _looks_like_atom_assignment(tok: str) -> bool:
    tok = tok.strip()
    return tok.startswith("A") and "=" in tok and tok[1:].split("=", 1)[0].isdigit()


def _looks_like_bond_assignment(tok: str) -> bool:
    """
    Heuristic for B<i,j>=code tokens.
    """
    tok = tok.strip()
    if not tok.startswith("B"):
        return False
    if "=" not in tok:
        return False
    lhs = tok[1:].split("=", 1)[0]
    if "," not in lhs:
        return False
    i, j = lhs.split(",", 1)
    return i.isdigit() and j.isdigit()


def _looks_like_group(tok: str) -> bool:
    tok = tok.strip()
    if not tok:
        return False
    if tok[0] not in ("a", "o", "&"):
        return False
    return ":" in tok


def export_curly_block_from_mol(
    mol: Chem.Mol,
    existing_meta: Optional[ChemAxonMeta] = None,
    index_base: int = 0,
    mode: str = "append",
) -> str:
    """
    Construct the contents of the { ... } block for a molecule.

    - existing_meta: if present, we can keep or replace its A/B/group tokens.
    - mode="append": keep existing tokens, just append new A/B/group tokens.
    - mode="replace": strip any existing A/B/group tokens first, then add fresh ones.

    Return value is the *inner* block, e.g. "A2=s;B1,3=e;&1:7,9".
    """
    # Fresh tokens from RDKit + CIP
    cip_tokens = export_cip_assignments(mol, index_base=index_base)
    ez_tokens = export_ez_assignments(mol, index_base=index_base)
    group_tokens = export_stereogroup_tokens(mol, index_base=index_base)

    if existing_meta is None:
        merged = cip_tokens + ez_tokens + group_tokens
        return ";".join(merged) if merged else ""

    orig_tokens = list(existing_meta.tokens) if existing_meta.tokens else []

    if mode == "replace":
        filtered_orig = []
        for tok in orig_tokens:
            if (
                _looks_like_atom_assignment(tok)
                or _looks_like_bond_assignment(tok)
                or _looks_like_group(tok)
            ):
                # drop old stereo encodings
                continue
            filtered_orig.append(tok)
        orig_tokens = filtered_orig

    merged = orig_tokens + cip_tokens + ez_tokens + group_tokens
    return ";".join(merged)


# -------------------------------------------------------------
# Convenience: export full enhanced SMILES
# -------------------------------------------------------------

def export_enhanced_smiles_from_mol(
    mol: Chem.Mol,
    existing_meta: Optional[ChemAxonMeta] = None,
    index_base: int = 0,
    mode: str = "append",
) -> str:
    """
    Return ChemAxon-style enhanced SMILES:

        <canonical SMILES> {A...;B...;a:...;o1:...;&1:...}

    with the standard single space before '{' if there is any metadata,
    or just plain SMILES if no tokens exist.
    """
    core = Chem.MolToSmiles(mol, canonical=True)
    block = export_curly_block_from_mol(
        mol, existing_meta=existing_meta, index_base=index_base, mode=mode
    )
    if not block:
        return core
    return f"{core} {{{block}}}"
