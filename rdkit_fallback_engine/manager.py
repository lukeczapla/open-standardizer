import rdkit
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from ..gpu.standardizer import gpu_standardize_once

class StandardizerFallbackEngine:
    """
    GPU → CPU fallback engine.
    If any GPU module fails (exceptions, NaNs, mismatched atom count),
    fall back to RDKit's MolStandardize pipeline.
    """

    def __init__(self):
        self.normalizer = rdMolStandardize.Normalizer()
        self.reionizer = rdMolStandardize.Reionizer()
        self.fragment_remover = rdMolStandardize.FragmentRemover()
        self.tautomer_enum = rdMolStandardize.TautomerEnumerator()

    def gpu_first(self, mol: Chem.Mol):
        """
        Try GPU pipeline → fallback to RDKit if needed.
        """
        try:
            gpu_out = gpu_standardize_once(mol)
            if gpu_out is None:
                raise RuntimeError("GPU returned None")
            if gpu_out.GetNumAtoms() != mol.GetNumAtoms():
                raise RuntimeError("GPU output atom mismatch")
            return gpu_out
        except Exception as e:
            return self.cpu_standardize(mol)

    def cpu_standardize(self, mol: Chem.Mol):
        """
        Equivalent to ChemAxon Standardizer → RDKit edition.
        """
        m = Chem.Mol(mol)

        # 1. Remove fragments
        m = self.fragment_remover(m)

        # 2. Normalize functional groups
        m = self.normalizer.normalize(m)

        # 3. Reionize according to rules
        m = self.reionizer.reionize(m)

        # 4. Tautomer canonicalization
        m = self.tautomer_enum.Canonicalize(m)

        return m

