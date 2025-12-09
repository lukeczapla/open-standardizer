import xml.etree.ElementTree as ET

from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.AllChem import EmbedMolecule
from rdkit.Chem.MolStandardize import rdMolStandardize


class OpenStandardizer:
    """
    Lightweight, pure-RDKit/MolVS standardizer driven directly
    by a ChemAxon-style XML config (like chemaxon-canon-stereo-keeplargest.xml).

    This is independent of the GPU/CPU policy layer – it's a
    CPU-only, “open” equivalent.
    """

    def __init__(self, config_file: str):
        self.tree = ET.parse(config_file)
        self.root = self.tree.getroot()

        # MolVS / rdMolStandardize primitives
        self.normalizer = rdMolStandardize.Normalizer()
        self.reionizer = rdMolStandardize.Reionizer()
        self.taut_enum = rdMolStandardize.TautomerEnumerator()
        self.frag_rem = rdMolStandardize.FragmentRemover()
        self.largest_frag = rdMolStandardize.LargestFragmentChooser()

    def apply(self, mol: Chem.Mol | None) -> Chem.Mol | None:
        if mol is None:
            return None

        m = Chem.Mol(mol)  # clone
        actions = self.root.find("Actions")
        if actions is None:
            return m

        for action in actions:
            tag = action.tag

            # ---- names mirroring your ChemAxon XML ----

            if tag == "ClearStereo":
                Chem.RemoveStereochemistry(m)

            elif tag == "RemoveFragment":
                # XML: Measure="heavyAtomCount" Method="keepLargest"
                # RDKit equivalent: keep largest fragment by heavy-atom count.
                m = self.largest_frag.choose(m)

            elif tag == "RemoveExplicitH":
                # Remove explicit H atoms (leave implicit hydrogen handling to RDKit)
                m = Chem.RemoveHs(m, implicitOnly=False)

            elif tag == "ClearIsotopes":
                for atom in m.GetAtoms():
                    atom.SetIsotope(0)

            elif tag == "Neutralize":
                # Reionizer canonicalizes charges / protonation state
                m = self.reionizer.reionize(m)

            elif tag == "Mesomerize":
                # There is no 1:1 ChemAxon mesomerizer in RDKit.
                # For now we just ensure stereochem/aromatic flags are consistent.
                Chem.rdmolops.AssignStereochemistry(m, cleanIt=True)

            elif tag == "Tautomerize":
                # MolVS canonical tautomer
                m = self.taut_enum.Canonicalize(m)

            elif tag == "Aromatize":
                # Kekulize + set aromaticity (close to ChemAxon “Aromatize general”)
                Chem.SanitizeMol(
                    m,
                    Chem.SANITIZE_KEKULIZE | Chem.SANITIZE_SETAROMATICITY,
                )

            elif tag == "RemoveAtomValues":
                # Clear atom mapping / R-labels / isotopes / explicit stereo flags
                for atom in m.GetAtoms():
                    atom.SetAtomMapNum(0)
                    if atom.HasProp("_MolFileRLabel"):
                        atom.ClearProp("_MolFileRLabel")
                    atom.SetIsotope(0)
                    atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)

            elif tag == "RemoveAttachedData":
                # Drop molecule-level props & name
                if m.HasProp("_Name"):
                    m.ClearProp("_Name")
                for prop in list(m.GetPropNames()):
                    m.ClearProp(prop)

            elif tag == "RemoveSgroups":
                # RDKit’s Sgroup exposure is limited; safest “no-op” placeholder.
                # You could call low-level APIs here if your RDKit build supports it.
                pass

            elif tag == "RemoveAtomMapping":
                for atom in m.GetAtoms():
                    atom.SetAtomMapNum(0)

            elif tag == "CleanGeometry":
                dim = action.get("dimension", "2D")
                if dim == "2D":
                    rdDepictor.Compute2DCoords(m)
                else:
                    EmbedMolecule(m)

            elif tag == "ClearCharges":
                # (bugfix: don’t return inside the loop)
                for atom in m.GetAtoms():
                    atom.SetFormalCharge(0)

            # If ChemAxon adds other tags, they safely fall through as no-ops.

        return m
