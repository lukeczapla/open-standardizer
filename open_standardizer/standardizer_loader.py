"""
ChemAxon XML-equivalent loader + dispatcher
Drop-in replacement for:
  biosignature.mol2mol.chemaxon.ChemaxonStandardisation(configurationName=...)
"""

import os
import xml.etree.ElementTree as ET
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize as molvs


BASE_DIR = os.path.join(os.path.dirname(__file__), "configs", "xml")


# ------------------------------------------------------------
# RDKit/MolVS primitive operations mapped 1:1 from XML actions
# ------------------------------------------------------------

def action_remove_largest_fragment(mol):
    lf = molvs.LargestFragmentChooser()
    return lf.choose(mol)

def action_clear_stereo(mol):
    mol = Chem.RemoveStereochemistry(mol)
    return mol

def action_remove_attached_data(mol):
    for atom in mol.GetAtoms():
        atom.ClearProp("molAtomMapNumber")
        for prop in atom.GetPropsAsDict().keys():
            try: atom.ClearProp(prop)
            except: pass
    return mol

def action_clear_atom_values(mol):
    # ChemAxon “Remove Atom Values” = clear map numbers, radicals, query flags
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
        atom.SetIsotope(0)
    return mol

def action_remove_explicit_h(mol):
    return Chem.RemoveHs(mol)

def action_clear_isotopes(mol):
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
    return mol

def action_neutralize(mol):
    un = molvs.Uncharger()
    return un.uncharge(mol)

def action_aromatize(mol):
    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
    return mol

def action_mesomerize(mol):
    # RDKit equivalent: canonical resonance structures → pick first
    res = Chem.ResonanceMolSupplier(mol, Chem.Kekulize(False))
    return Chem.Mol(res[0]) if len(res) > 0 else mol

def action_tautomerize(mol):
    te = molvs.TautomerEnumerator()
    return te.Canonicalize(mol)

def action_clean_2d(mol):
    Chem.rdDepictor.Compute2DCoords(mol)
    return mol


# ------------------------------------------------------------
# Action dispatcher mapping the XML tags → RDKit functions
# ------------------------------------------------------------

ACTION_MAP = {
    "RemoveFragment": action_remove_largest_fragment,
    "ClearStereo": action_clear_stereo,
    "RemoveAttachedData": action_remove_attached_data,
    "ClearAtomValues": action_clear_atom_values,
    "RemoveExplicitH": action_remove_explicit_h,
    "ClearIsotopes": action_clear_isotopes,
    "Neutralize": action_neutralize,
    "Aromatize": action_aromatize,
    "Mesomerize": action_mesomerize,
    "Tautomerize": action_tautomerize,
    "CleanGeometry": action_clean_2d,
}


# ------------------------------------------------------------
# Parse ChemAxon-equivalent XML → list of RDKit actions
# ------------------------------------------------------------

def load_xml_actions(configuration_name: str):
    xml_path = os.path.join(BASE_DIR, f"{configuration_name}.xml")
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Missing XML: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    actions = []
    for action in root.find("Actions"):
        tag = action.tag
        if tag not in ACTION_MAP:
            print(f"[WARN] No RDKit mapping for {tag}, skipping")
            continue
        actions.append(ACTION_MAP[tag])

    return actions


# ------------------------------------------------------------
# Public entrypoint: standardize(smiles, configName)
# ------------------------------------------------------------

def standardize_smiles(smiles: str, configuration_name: str) -> str:
    use_gpu = (config.get("use_gpu") is True)

    if use_gpu:
        try:
            from .gpu.standardize_gpu_old import gpu_standardize
            return gpu_standardize(smiles)
        except Exception:
            pass
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError(f"Invalid SMILES: {smiles}")

    actions = load_xml_actions(configuration_name)

    for fn in actions:
        try:
            mol = fn(mol)
        except Exception as e:
            print(f"[WARN] action {fn.__name__} failed: {e}")

    return Chem.MolToSmiles(mol, canonical=True)
