# open-standardizer

To be re-written later, it is for molecules (e.g. SMILES), reads ChemAxon-compatible XML configuration to standardize molecules.
The format of the Enhanced SMILES is very similar to the CXSMILES/CXSMARTS formats, but may not be 100% ChemAxon compatible yet.
Rdkit (with MolVS) is used in place of a tool like ChemAxon (Java) for standardizing input molecules provided in a SMILES format, with some experimental CUDA code for GPU acceleration.

Available through a non-copyleft open source license, BSD-3 clause

## Installation

```

  pip install .

  # to install the project as a dynamic link for live edits:
  pip install -e .

  # to compile GPU code into the system and emit libraries for your platform (CUDA nvcc and NVIDIA drivers required):
  CMAKE_ARGS="-DOPEN_STANDARDIZER_ENABLE_CUDA=ON" pip install .

```

## Examples

Running tests:

```

  # Usage:
  python open_standardizer/tests/standardize_cli.py --help


  python open_standardizer/tests/standardize_cli.py --xml-config chemaxon-canon-nostereo.xml input.csv > output.csv

```

Python code (using API directly with `rdkit` for CPU):

```

from rdkit import Chem

from open_standardizer.cpu_ops import op_mesomerize

def test(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    print("INPUT:      ", smiles)
    print("CANONICAL:  ", Chem.MolToSmiles(mol))

    out_mol = op_mesomerize(Chem.Mol(mol))
    print("MESO SMILES:", Chem.MolToSmiles(out_mol))
    print("-" * 50)


if __name__ == "__main__":
    # nitrobenzene in a non-canonical form
    test("[O-][N+](=O)c1ccccc1")

    # benzoate anion in a non-canonical form
    test("[O-]C(=O)c1ccccc1")

```

