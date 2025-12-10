from __future__ import annotations
from typing import List, Tuple
from rdkit import Chem
from rdkit.Chem import rdchem, rdmolops


# ------------------------------
# Atom and bond ranking helpers
# ------------------------------

def _atom_rank(atom: Chem.Atom, mol: Chem.Mol) -> Tuple:
    """
    Heuristic ChemAxon-like atom ranking.

    Lower tuples are "better" and are visited earlier.

    Priorities (in order):
      1) terminal atoms first (degree == 1)
      2) hetero before carbon
      3) heavier elements first
      4) aromatic before aliphatic
      5) ring atoms before acyclic
      6) smaller rings before larger rings
      7) more substituted (higher degree)
      8) more hetero neighbors
      9) sp > sp2 > sp3 for linear / unsaturated fragments
     10) lower |formal charge|
     11) higher valence
     12) original index
    """
    idx = atom.GetIdx()
    at_num = atom.GetAtomicNum()
    degree = atom.GetDegree()

    # 1) terminal vs internal
    is_terminal = 0 if degree == 1 else 1

    # 2) hetero vs carbon
    is_hetero = 0 if (at_num != 6 and at_num > 1) else 1  # 0 = hetero, 1 = carbon/other

    # 3) heavier element first (invert sign)
    # 4) aromatic
    is_aromatic = 0 if atom.GetIsAromatic() else 1

    # 5) ring / acyclic
    in_ring = 0 if atom.IsInRing() else 1

    # 6) smallest ring size containing this atom
    ring_info = mol.GetRingInfo()
    atom_rings = [r for r in ring_info.AtomRings() if idx in r]
    smallest_ring = min((len(r) for r in atom_rings), default=99)

    # 7) substitution (more substituted first)
    degree_score = -degree

    # 8) hetero neighbors count (more hetero neighbors → better)
    hetero_neighbors = 0
    for nbr in atom.GetNeighbors():
        z = nbr.GetAtomicNum()
        if z != 6 and z > 1:
            hetero_neighbors += 1
    hetero_neighbor_score = -hetero_neighbors

    # 9) hybridization: prefer sp (for C≡C), then sp2, then sp3/other
    hyb = atom.GetHybridization()
    if hyb == rdchem.HybridizationType.SP:
        hyb_rank = 0
    elif hyb == rdchem.HybridizationType.SP2:
        hyb_rank = 1
    else:
        hyb_rank = 2

    # 10) charge magnitude
    formal_charge = abs(atom.GetFormalCharge())

    # 11) valence-ish preference
    total_valence = -(atom.GetTotalValence() or degree)

    # 12) original index as last tie-breaker
    return (
        is_terminal,
        is_hetero,
        -at_num,
        is_aromatic,
        in_ring,
        smallest_ring,
        degree_score,
        hetero_neighbor_score,
        hyb_rank,
        formal_charge,
        total_valence,
        idx,
    )


def _bond_rank(bond: Chem.Bond) -> Tuple:
    """
    Rank bonds for traversal ordering.
    Lower tuple = higher priority.
    """
    btype = bond.GetBondType()
    # approximate strength ordering:
    if btype == rdchem.BondType.TRIPLE:
        bt = 0
    elif btype == rdchem.BondType.DOUBLE:
        bt = 1
    elif btype == rdchem.BondType.AROMATIC:
        bt = 2
    else:
        bt = 3  # single, others

    in_ring = 0 if bond.IsInRing() else 1
    stereo = 0 if bond.GetStereo() != rdchem.BondStereo.STEREONONE else 1

    return (bt, in_ring, stereo)


# ------------------------------
# Per-component ordering
# ------------------------------

def _chemaxon_like_component_order(
    mol: Chem.Mol,
    start_idx: int,
    atom_ranks: dict[int, Tuple],
) -> List[int]:
    """
    Depth-first traversal from start_idx, ordering neighbors by:
      - bond rank
      - atom rank
    """
    visited = set()
    order: List[int] = []
    stack: List[int] = [start_idx]

    while stack:
        idx = stack.pop()
        if idx in visited:
            continue
        visited.add(idx)
        order.append(idx)

        atom = mol.GetAtomWithIdx(idx)
        nbrs = []
        for bond in atom.GetBonds():
            j = bond.GetOtherAtomIdx(idx)
            if j in visited:
                continue
            nbrs.append((_bond_rank(bond), atom_ranks[j], j))

        # sort reverse so “best” neighbor is popped first
        nbrs.sort(reverse=True)
        for _b_rank, _a_rank, j in nbrs:
            stack.append(j)

    return order


def _chemaxon_like_global_order(mol: Chem.Mol) -> List[int]:
    """
    Global atom order for the whole molecule.

      - Compute heuristic rank per atom.
      - Identify connected components.
      - Sort components by:
          * heavy atom count (descending)
          * then best atom_rank in component (ascending)
      - For each component:
          * pick best atom as root
          * DFS with _chemaxon_like_component_order
    """
    n = mol.GetNumAtoms()
    if n == 0:
        return []

    # Per-atom ranks once
    atom_ranks = {
        a.GetIdx(): _atom_rank(a, mol)
        for a in mol.GetAtoms()
    }

    # Connected components
    comps = rdmolops.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
    comp_info = []
    for comp_idx, atom_ids in enumerate(comps):
        heavy_count = sum(
            1 for i in atom_ids if mol.GetAtomWithIdx(i).GetAtomicNum() > 1
        )
        best_atom = min(atom_ids, key=lambda i: atom_ranks[i])
        best_rank = atom_ranks[best_atom]
        comp_info.append(
            (comp_idx, heavy_count, best_atom, best_rank, list(atom_ids))
        )

    # Sort components: more heavy atoms first, then best_rank
    comp_info.sort(key=lambda x: (-x[1], x[3]))

    global_order: List[int] = []
    seen = set()

    for _idx, _heavy, root, _rank, atom_ids in comp_info:
        comp_order = _chemaxon_like_component_order(mol, root, atom_ranks)
        for idx in comp_order:
            if idx in atom_ids and idx not in seen:
                seen.add(idx)
                global_order.append(idx)

    return global_order


# ------------------------------
# Public API
# ------------------------------

def chemaxon_like_reorder(mol: Chem.Mol) -> Chem.Mol:
    """
    Return a *new* molecule whose atom indices are in a ChemAxon-like
    canonical order.
    """
    if mol is None:
        return None

    order = _chemaxon_like_global_order(mol)
    if not order:
        return mol

    # RenumberAtoms expects new->old mapping; 'order' is exactly that.
    return Chem.RenumberAtoms(mol, order)


def chemaxon_like_smiles_from_mol(mol: Chem.Mol) -> str:
    """
    Produce a deterministic ChemAxon-style SMILES:

      - Reorder atoms with our heuristic canonicalizer.
      - Use canonical=False so RDKit respects atom order.
    """
    if mol is None:
        return ""

    rmol = chemaxon_like_reorder(mol)
    return Chem.MolToSmiles(rmol, canonical=False, isomericSmiles=True)


def chemaxon_like_smiles(smiles: str) -> str:
    """
    Convenience: SMILES → Mol → ChemAxon-like SMILES.
    """
    if not smiles:
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles  # fallback to original
    return chemaxon_like_smiles_from_mol(mol)
