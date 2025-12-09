
# RDKit CPU fallback module (trimmed scaffold)
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

def cpu_standardize(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""

    # Largest fragment
    lf = rdMolStandardize.LargestFragmentChooser()
    mol = lf.choose(mol)

    # Cleanup + normalize
    cleaner = rdMolStandardize.Cleanup()
    mol = cleaner.clean(mol)

    # Tautomer canonicalization
    te = rdMolStandardize.TautomerEnumerator()
    mol = te.Canonicalize(mol)

    return Chem.MolToSmiles(mol, canonical=True)
