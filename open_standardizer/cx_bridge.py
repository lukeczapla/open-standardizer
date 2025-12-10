# open_standardizer/cx_bridge.py
from __future__ import annotations
import sys

from typing import Optional, Tuple
from rdkit import Chem

try:
    # RDKit’s CXSMILES writer; exact location can vary a bit by version
    from rdkit.Chem.rdmolfiles import MolToCXSmiles  # type: ignore[attr-defined]
except ImportError:
    sys.stderr.write("\nrdkit.Chem.MolToCXSmiles was not available in Rdkit\n")
    sys.stderr.flush()
    MolToCXSmiles = None  # fall back gracefully


def parse_smiles_core_and_pipe(s: str) -> Tuple[str, Optional[str]]:
    """
    Split 'SMILES|...|' into (core_smiles, pipe_block_without_bars).

    Examples:
        "CCO"                        -> ("CCO", None)
        "CCO |$foo$|"                -> ("CCO", "$foo$")
        "c1ccccc1C |a:1,2,3,w:1.0|"  -> ("c1ccccc1C", "a:1,2,3,w:1.0")
    """
    s = s.strip()
    first_bar = s.find("|")
    if first_bar < 0:
        return s, None

    last_bar = s.rfind("|")
    if last_bar <= first_bar:
        # malformed; treat whole thing as SMILES
        return s, None

    core = s[:first_bar].rstrip()
    block = s[first_bar + 1:last_bar]
    return core, block.strip() or None


def mol_from_smiles_or_cx(smiles: str) -> Optional[Chem.Mol]:
    """
    RDKit’s MolFromSmiles already accepts its own CXSMILES dialect.
    This helper just wraps that and strips leading/trailing whitespace.
    """
    return Chem.MolFromSmiles(smiles.strip())


def mol_to_cxsmiles(mol: Chem.Mol) -> str:
    """
    Prefer RDKit’s CXSMILES writer if available; otherwise, fall back
    to plain canonical SMILES.
    """
    if MolToCXSmiles is not None:
        return MolToCXSmiles(mol)
    return Chem.MolToSmiles(mol, canonical=True)
