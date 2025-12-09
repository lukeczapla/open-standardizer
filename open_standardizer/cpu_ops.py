from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops, rdchem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumerator, CleanupParameters
from .enhanced_smiles import enhanced_smiles_to_mol

#from rdkit import RDLogger

#RDLogger.DisableLog("rdApp.warning")

_STRIPPER_DEFAULT_KEEP_LAST = rdMolStandardize.FragmentRemover()

# ---- Tautomer enumerator with higher limits ----
_TAUTOMER_PARAMS = CleanupParameters()
_TAUTOMER_PARAMS.maxTransforms = 5000
_TAUTOMER_PARAMS.maxTautomers = 2000

_TAUTOMER_ENUMERATOR = TautomerEnumerator(_TAUTOMER_PARAMS)


def _ensure_mol(mol):
    """
    Accept either RDKit Mol or SMILES string.
    Always return a *copy* of a Mol or None on failure.
    """
    if isinstance(mol, Chem.Mol):
        return Chem.Mol(mol)
    if isinstance(mol, str):
        return Chem.MolFromSmiles(mol)
    return None


def _strip_with_remover(remover, mol, allow_empty: bool):
    """
    Helper to apply a FragmentRemover but optionally prevent
    ending up with an empty / zero-atom mol.
    """
    mol = _ensure_mol(mol)
    if mol is None:
        return None
    try:
        out = remover(mol)
        if out is None or out.GetNumAtoms() == 0:
            # RDKit can't represent empty; keep original
            return mol if not allow_empty else mol
        return out
    except Exception:
        return mol

# ---------- 1. CLEAR STEREO ----------
def op_clear_stereo(mol):
    mol = _ensure_mol(mol)
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    for atom in mol.GetAtoms():
        atom.SetChiralTag(rdchem.ChiralType.CHI_UNSPECIFIED)
    return mol


# ---------- 2. REMOVE FRAGMENT (KEEP LARGEST) ----------
def op_remove_fragment_keeplargest(mol):
    mol = _ensure_mol(mol)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if not frags:
        return mol
    largest = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    try:
        Chem.SanitizeMol(largest)
    except Exception:
        # If sanitize fails, just return the original molecule
        return mol
    return largest


# ---------- 3. REMOVE ATTACHED DATA ----------
def op_remove_attached_data(mol):
    mol = _ensure_mol(mol)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.ClearProp("")
        for k in atom.GetPropsAsDict().keys():
            atom.ClearProp(k)
    for bond in mol.GetBonds():
        for k in bond.GetPropsAsDict().keys():
            bond.ClearProp(k)
    return mol


# ---------- 4. REMOVE ATOM VALUES ----------
def op_remove_atom_values(mol):
    mol = _ensure_mol(mol)
    if mol is None:
        return None    
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
        atom.SetFormalCharge(atom.GetFormalCharge())  # normalize
    return mol


# ---------- 5. REMOVE EXPLICIT HYDROGENS ----------
def op_remove_explicit_h(mol):
    mol = _ensure_mol(mol)
    if mol is None:
        return None
    return Chem.RemoveHs(mol, updateExplicitCount=True)


# ---------- 6. CLEAR ISOTOPES ----------
def op_clear_isotopes(mol):
    mol = _ensure_mol(mol)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
    return mol


# ---------- 7. NEUTRALIZE ----------
def _neutralize(mol):
    mol = _ensure_mol(mol)
    if mol is None:
        return None
    patterns = (
        ("[n+;H]", "n"),
        ("[N+;!H0]", "N"),
        ("[$([O-]);!$([O-][#7])]", "O"),
        ("[S-]", "S"),
        ("[$([N-]);!$([N-][#6]);!$([N-][#7])]", "N"),
    )
    replaced = False

    for smarts, repl in patterns:
        while True:
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
            if not matches:
                break
            replaced = True
            for atom_idx in matches:
                atom = mol.GetAtomWithIdx(atom_idx[0])
                atom.SetFormalCharge(0)

    if replaced:
        Chem.SanitizeMol(mol)
    return mol


def op_neutralize(mol):
    try:
        return _neutralize(mol)
    except Exception:
        return mol


# ---------- 8. MESOMERIZE (CHEMAXON-LIKE RESONANCE NORMALIZATION) ----------
def op_mesomerize(mol):
    """
    Pick a canonical resonance form.

    Uses Chem.ResonanceMolSupplier when available.
    If that API is missing in this RDKit build, this op is a no-op.
    """
    mol = _ensure_mol(mol)
    if mol is None:
        return None
    try:
        res = rdchem.ResonanceMolSupplier(mol, rdchem.UNCONSTRAINED_ANIONS)
    except Exception:
        return mol

    if len(res) > 0:
        best = Chem.Mol(res[0])
        try:
            Chem.SanitizeMol(best)
        except Exception:
            return mol
        return best
    return mol


# ---------- 9. TAUTOMERIZE ----------
def op_tautomerize(mol):
    """
    Tautomer normalization using a global TautomerEnumerator
    with higher maxTransforms / maxTautomers.
    """
    mol = _ensure_mol(mol)
    if mol is None:
        return None
    try:
        best = _TAUTOMER_ENUMERATOR.Canonicalize(mol)
    except Exception:
        # if RDKit freaks out (still hits a limit, weird structure, etc.)
        return mol

    # Defensive: don’t propagate None or empty mols
    if best is None or best.GetNumAtoms() == 0:
        return mol

    return best


# ---------- 10. AROMATIZE ----------
def op_aromatize(mol):
    mol = _ensure_mol(mol)
    if mol is None:
        return None
    Chem.SetAromaticity(mol)
    AllChem.SetAromaticity(mol)
    return mol


# ---------- 11. REMOVE FRAGMENT SMALLEST ----------
def op_remove_fragment_smallest(mol):
    """
    Keep all fragments except the one with the *fewest* heavy atoms.
    Simple "remove smallest salt" style behavior.
    """
    mol = _ensure_mol(mol)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if not frags:
        return mol

    # heavy atom counts
    with_counts = [(f, f.GetNumHeavyAtoms()) for f in frags]
    # sort by heavy atom count ascending
    with_counts.sort(key=lambda x: x[1])

    # drop the smallest, keep all others with >0 heavy atoms
    kept = [f for f, cnt in with_counts[1:] if cnt > 0]
    if not kept:
        # if everything is tiny, just keep original
        return mol

    combo = kept[0]
    for f in kept[1:]:
        combo = rdmolops.CombineMols(combo, f)

    Chem.SanitizeMol(combo)
    return combo


# usedefaultsalts="true", dontremovelastcomponent="true"
def op_strip_salts_default_keep_last(mol):
    return _strip_with_remover(_STRIPPER_DEFAULT_KEEP_LAST, mol, allow_empty=False)


# usedefaultsalts="true", dontremovelastcomponent="false"
# We approximate by **allowing all salts to be stripped**, but if RDKit
# returns an empty / broken mol we fall back to original.
def op_strip_salts_default_allow_empty(mol):
    return _strip_with_remover(_STRIPPER_DEFAULT_KEEP_LAST, mol, allow_empty=True)


# usedefaultsalts="false", dontremovelastcomponent="true"
# TODO: wire custom salt SMARTS here. For now we just use the same default
# remover but keep_last semantics.
def op_strip_salts_custom_keep_last(mol):
    # placeholder: behaves like default_keep_last until you add custom patterns
    return _strip_with_remover(_STRIPPER_DEFAULT_KEEP_LAST, mol, allow_empty=False)


# usedefaultsalts="false", dontremovelastcomponent="false"
def op_strip_salts_custom_allow_empty(mol):
    # placeholder: behaves like default_allow_empty until you add custom patterns
    return _strip_with_remover(_STRIPPER_DEFAULT_KEEP_LAST, mol, allow_empty=True)

# ---------- 12. CLEAN GEOMETRY ----------
def op_clean_geometry(mol):
    """
    ChemAxon CleanGeometry analogue.

    For SMILES-based standardization this is effectively a no-op on the
    connectivity (bonds/atoms). We do a best-effort 2D coordinate cleanup
    on a copy and return it.
    """
    m = _ensure_mol(mol)
    if m is None:
        return None
    try:
        # This only affects coordinates, not bonding
        AllChem.Compute2DCoords(m)
        return m
    except Exception:
        # If coordinate generation fails for any reason,
        # fall back to the original molecule
        return mol


# --------- MAPPING TO ACTION NAMES ---------
CPU_OPS = {
    # stereo
    "clear_stereo": op_clear_stereo,
    "ClearStereo": op_clear_stereo,  # ChemAxon-style name (just in case)

    # fragments
    "remove_fragment_keeplargest": op_remove_fragment_keeplargest,
    "remove_largest_fragment": op_remove_fragment_keeplargest,  # alias
    "RemoveLargestFragment": op_remove_fragment_keeplargest,    # ChemAxon-style

    "remove_fragment_smallest": op_remove_fragment_smallest,

    # salts
    "strip_salts_default_keep_last": op_strip_salts_default_keep_last,
    "strip_salts_default_allow_empty": op_strip_salts_default_allow_empty,
    "strip_salts_custom_keep_last": op_strip_salts_custom_keep_last,
    "strip_salts_custom_allow_empty": op_strip_salts_custom_allow_empty,

    # attached data / atom values
    "remove_attached_data": op_remove_attached_data,
    "RemoveAttachedData": op_remove_attached_data,  # ChemAxon-style

    "remove_atom_values": op_remove_atom_values,
    "ClearAtomValues": op_remove_atom_values,       # <-- this is the one you’re missing

    # hydrogens / isotopes
    "remove_explicit_h": op_remove_explicit_h,
    "RemoveExplicitH": op_remove_explicit_h,        # ChemAxon-style

    "clear_isotopes": op_clear_isotopes,
    "ClearIsotopes": op_clear_isotopes,             # ChemAxon-style

    # charge / resonance / tautomers / aromaticity
    "neutralize": op_neutralize,
    "Neutralize": op_neutralize,                    # ChemAxon-style

    "mesomerize": op_mesomerize,
    "Mesomerize": op_mesomerize,                    # ChemAxon-style

    "tautomerize": op_tautomerize,
    "Tautomerize": op_tautomerize,                  # ChemAxon-style

    "aromatize": op_aromatize,
    "Aromatize": op_aromatize,                      # ChemAxon-style

    # --- CleanGeometry mapping ---
    "clean_geometry": op_clean_geometry,  # internal/friendly name
    "CleanGeometry": op_clean_geometry,   # ChemAxon XML action name
}


def cpu_execute(op_name, mol):
    """
    Executes a CPU operation by name.
    Returns the mol (to be converted to SMILES or other format later)
    """
    fn = CPU_OPS.get(op_name)
    if mol is None:
        return None

    # Normalize input → work_mol is always a Chem.Mol or None
    if isinstance(mol, Chem.Mol):
        work_mol = Chem.Mol(mol)  # copy
        orig_smiles = Chem.MolToSmiles(mol)
    else:
        # caller passed a SMILES string
        orig_smiles = str(mol)
        work_mol, parsed = enhanced_smiles_to_mol(orig_smiles)

    if work_mol is None:
        # Can't even parse → return original mol
        return mol

    if fn is None:
        # Unknown op → return original SMILES
        return mol
    try:
        new_mol = fn(work_mol)
    except Exception:
        # Any RDKit craziness during the op → fall back to original
        return work_mol

    # Some ops might (incorrectly) return None: treat as no change
    if new_mol is None:
        new_mol = work_mol

    return new_mol

