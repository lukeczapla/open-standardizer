from __future__ import annotations

from typing import List, Optional, Union

from rdkit import Chem

from .xml_loader import load_xml_actions
from .gpu_cpu_policy_manager import (
    GPU_CPU_MANAGER,
    Policy,
    try_gpu_standardize,
)
from .cpu_ops import cpu_execute, CPU_OPS
from .batch_engine import BatchStandardizer
from .stereo_export import export_curly_block_from_mol, export_enhanced_smiles_from_mol

from .enhanced_smiles import enhanced_smiles_to_mol, ChemAxonMeta, parse_chemaxon_enhanced

# ChemAxon-equivalent canonicalization pipeline (XML-matched)
DEFAULT_OPS: List[str] = [
    "clear_stereo",
    "strip_salts_default_keep_last",
    "remove_largest_fragment",
    "remove_attached_data",
    "remove_atom_values",
    "remove_explicit_h",
    "clear_isotopes",
    "neutralize",
    "mesomerize",
    "tautomerize",
    "aromatize",
]


class Standardizer:
    """
    Orchestrates standardization:

      - loads ChemAxon-style XML configs
      - applies actions in order
      - tries GPU first per operation
      - falls back to RDKit/MolVS CPU ops

    NOTE:
      This class itself operates on *plain SMILES / Mol*.
      Enhanced SMILES (with '{...}') should go through the
      helper wrappers below (standardize_smiles / standardize).
    """

    def __init__(
        self,
        xml_config_path: Optional[str] = None,
        policy: Optional[Policy] = None,
    ) -> None:
        self.xml_config_path = xml_config_path
        self.policy: Policy = policy or GPU_CPU_MANAGER

        self.actions: List[dict] = []
        if xml_config_path:
            self.actions = load_xml_actions(xml_config_path)

    def set_xml(self, xml_path: str) -> None:
        self.xml_config_path = xml_path
        self.actions = load_xml_actions(xml_path)

    def _apply_op(self, op_name: str, smiles: str) -> Optional[str]:
        """
        Legacy SMILES-based single-op helper.

        Kept for backwards-compatibility; the main code paths now use
        a Mol-based pipeline (_apply_ops_to_mol) to avoid repeated
        SMILES↔Mol conversions.
        """
        # 1. GPU path
        gpu_fn = self.policy.try_gpu(op_name)
        if gpu_fn is not None:
            gpu_out = try_gpu_standardize(smiles, [op_name], self.policy, gpu_fn)
            if isinstance(gpu_out, str):
                return gpu_out

        # 2. CPU fallback: use Mol-based cpu_execute and convert back to SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        new_mol = cpu_execute(op_name, mol)
        if new_mol is None:
            return None

        return Chem.MolToSmiles(new_mol)

    def standardize(self, mol: Optional[Chem.Mol]) -> Optional[Chem.Mol]:
        """
        Run the full XML-driven pipeline on a single RDKit Mol.

        IMPORTANT:
          - This function does *not* know about ChemAxon curly metadata.
          - Use the string-based helpers for "enhanced SMILES in / out".
        """
        if mol is None:
            return None

        if not self.actions:
            # Nothing to do, return a copy
            return Chem.Mol(mol)

        ops = [entry["op"] for entry in self.actions]
        out_mol = _apply_ops_to_mol(mol, ops, self.policy)
        return out_mol


def _run_cpu_op_on_mol(
    op_name: str,
    mol: Chem.Mol,
    policy: Policy,
) -> Chem.Mol:
    """
    Run a single CPU operation on a Mol (via cpu_execute: Mol -> Mol).
    """
    # Always operate on a copy at the cpu_execute layer
    new_mol = cpu_execute(op_name, mol)
    if new_mol is None or new_mol.GetNumAtoms() == 0:
        return mol
    return new_mol


def _apply_ops_to_mol(
    mol: Chem.Mol,
    ops: List[str],
    policy: Policy,
) -> Optional[Chem.Mol]:
    """
    Efficient Mol→Mol pipeline:

      - Keeps a Mol in memory for all CPU ops.
      - Only converts to/from SMILES when going through a GPU kernel.
    """
    current_mol = Chem.Mol(mol)  # work on a copy

    for op_name in ops:
        # 1) GPU attempt (works on SMILES)
        gpu_fn = policy.try_gpu(op_name)
        if gpu_fn is not None:
            current_smiles = Chem.MolToSmiles(current_mol)
            gpu_out = try_gpu_standardize(current_smiles, [op_name], policy, gpu_fn)
            if isinstance(gpu_out, str):
                new_mol = Chem.MolFromSmiles(gpu_out)
                if new_mol is None:
                    return None
                current_mol = new_mol
                continue

        # 2) CPU fallback (Mol-only)
        current_mol = _run_cpu_op_on_mol(op_name, current_mol, policy)

    return current_mol


def _apply_ops_to_smiles(
    smiles: str,
    ops: List[str],
    policy: Policy,
) -> Optional[str]:
    """
    Core SMILES → SMILES transformation loop.

    Now implemented as:

        SMILES → Mol (once) → ops on Mol → SMILES (once)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    out_mol = _apply_ops_to_mol(mol, ops, policy)
    if out_mol is None:
        return None

    return Chem.MolToSmiles(out_mol)


class BatchStandardizerWrapper:
    """
    (Optional) Wrapper if you ever want to expose a pure SMILES interface
    around BatchStandardizer. Kept here for clarity; your existing
    BatchStandardizer is already used by Standardizer.standardize_many.
    """
    pass


def standardize_smiles(
    smiles: str,
    xml_path: str,
    preserve_chemaxon_meta: bool = True,
) -> Optional[str]:
    """
    Convenience wrapper: load XML, standardize a single *enhanced* SMILES
    and return SMILES.
    """
    parsed = parse_chemaxon_enhanced(smiles)
    core = parsed.core_smiles

    mol = Chem.MolFromSmiles(core)
    if mol is None:
        return None

    std = Standardizer(xml_path)
    out_mol = std.standardize(mol)
    if out_mol is None:
        return None

    core_out = Chem.MolToSmiles(out_mol)

    if preserve_chemaxon_meta and parsed.meta:
        return f"{core_out} {{{parsed.meta.to_raw()}}}"

    return core_out


def standardize_many(
    smiles_list: List[str],
    ops: Optional[List[str]] = None,
) -> List[Optional[str]]:
    """
    Batch standardization using DEFAULT_OPS and the global GPU/CPU manager.
    """
    ops = ops or DEFAULT_OPS

    # Parse enhanced → core for the batch
    parsed_list = [parse_chemaxon_enhanced(s) for s in smiles_list]
    core_list = [p.core_smiles for p in parsed_list]

    engine = BatchStandardizer(GPU_CPU_MANAGER)
    core_out_list = engine.run(core_list, ops)

    results: List[Optional[str]] = []
    for parsed, core_out in zip(parsed_list, core_out_list):
        if core_out is None:
            results.append(None)
            continue

        if parsed.meta:
            results.append(f"{core_out} {{{parsed.meta.to_raw()}}}")
        else:
            results.append(core_out)

    return results


def standardize(
    smiles: str,
    ops: List[str] = DEFAULT_OPS,
    policy: Policy = GPU_CPU_MANAGER,
    preserve_chemaxon_meta: bool = True,
    regenerate_chemaxon_meta: bool = False,
    index_base: int = 0,
) -> Optional[str]:
    """
    Functional API (no XML), *enhanced SMILES aware*.
    """
    mol, meta = enhanced_smiles_to_mol(smiles, index_base=index_base)
    if mol is None:
        return None

    out_mol = _apply_ops_to_mol(mol, ops, policy)
    if out_mol is None:
        return None

    current = Chem.MolToSmiles(out_mol)

    if regenerate_chemaxon_meta:
        new_block = export_curly_block_from_mol(
            out_mol, existing_meta=meta, index_base=index_base
        )
        if new_block:
            return f"{current} {{{new_block}}}"
        return current

    if preserve_chemaxon_meta and meta and meta.raw.strip():
        return f"{current} {{{meta.to_raw()}}}"

    return current


def standardize_mol(
    mol: Optional[Chem.Mol],
    ops: List[str] = DEFAULT_OPS,
    policy: Policy = GPU_CPU_MANAGER,
) -> Optional[Chem.Mol]:
    """
    Functional Mol→Mol API (no XML, no ChemAxon curly metadata).
    """
    if mol is None:
        return None

    out_mol = _apply_ops_to_mol(mol, ops, policy)
    return out_mol


def standardize_mol_xml(
    mol: Optional[Chem.Mol],
    xml_path: str,
    policy: Policy = GPU_CPU_MANAGER,
) -> Optional[Chem.Mol]:
    if mol is None:
        return None
    std = Standardizer(xml_path, policy=policy)
    return std.standardize(mol)


def standardize_molblock(
    molblock: str,
    ops: List[str] = DEFAULT_OPS,
    policy: Policy = GPU_CPU_MANAGER,
    return_molblock: bool = True,
    v3000: bool = False,
) -> Optional[str]:
    """
    Standardize a molfile (MolBlock string) and return either
    a new MolBlock (default) or a canonical SMILES.
    """
    if not molblock.strip():
        return None

    mol = Chem.MolFromMolBlock(molblock, sanitize=True)
    if mol is None:
        return None

    out_mol = standardize_mol(mol, ops=ops, policy=policy)
    if out_mol is None:
        return None

    if return_molblock:
        return Chem.MolToMolBlock(out_mol, forceV3000=v3000)

    return Chem.MolToSmiles(out_mol)


def standardize_enhanced_smiles_to_molblock(
    smiles: str,
    ops: List[str] = DEFAULT_OPS,
    policy: Policy = GPU_CPU_MANAGER,
    index_base: int = 0,
    v3000: bool = False,
) -> Optional[str]:
    """
    Enhanced SMILES → standardized molfile.
    """
    mol, _meta = enhanced_smiles_to_mol(smiles, index_base=index_base)
    if mol is None:
        return None

    out_mol = standardize_mol(mol, ops=ops, policy=policy)
    if out_mol is None:
        return None

    return Chem.MolToMolBlock(out_mol, forceV3000=v3000)


MoleculeLike = Union[str, Chem.Mol]


def standardize_any(
    x: MoleculeLike,
    ops: List[str] = DEFAULT_OPS,
    policy: Policy = GPU_CPU_MANAGER,
    assume_enhanced_smiles: bool = False,
    index_base: int = 0,
    output_format: str = "smiles",  # "smiles", "mol", "molblock"
) -> Optional[Union[str, Chem.Mol]]:
    """
    Convenience front door.
    """
    # 1) Already a Mol
    if isinstance(x, Chem.Mol):
        out_mol = standardize_mol(x, ops=ops, policy=policy)
        if out_mol is None:
            return None
        if output_format == "mol":
            return out_mol
        if output_format == "molblock":
            return Chem.MolToMolBlock(out_mol)
        return Chem.MolToSmiles(out_mol)

    # 2) String input
    s = x.strip()
    if not s:
        return None

    # crude molfile sniff
    if "\n" in s and "M  END" in s:
        mol = Chem.MolFromMolBlock(s, sanitize=True)
        if mol is None:
            return None
        out_mol = standardize_mol(mol, ops=ops, policy=policy)
        if out_mol is None:
            return None
        if output_format == "mol":
            return out_mol
        if output_format == "molblock":
            return Chem.MolToMolBlock(out_mol)
        return Chem.MolToSmiles(out_mol)

    # 3) SMILES / enhanced SMILES
    is_enhanced = assume_enhanced_smiles or ("{" in s and "}" in s)
    if is_enhanced:
        out = standardize(
            s,
            ops=ops,
            policy=policy,
            preserve_chemaxon_meta=True,
            regenerate_chemaxon_meta=False,
            index_base=index_base,
        )
        if out is None:
            return None
        if output_format == "smiles":
            return out
        # parse back to Mol if needed
        mol, _meta = enhanced_smiles_to_mol(out, index_base=index_base)
        if mol is None:
            return None
        if output_format == "mol":
            return mol
        if output_format == "molblock":
            return Chem.MolToMolBlock(mol)
        return out

    # 4) plain SMILES (no curly block)
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None

    out_mol = standardize_mol(mol, ops=ops, policy=policy)
    if out_mol is None:
        return None

    if output_format == "smiles":
        return Chem.MolToSmiles(out_mol)
    if output_format == "mol":
        return out_mol
    if output_format == "molblock":
        return Chem.MolToMolBlock(out_mol)
    return Chem.MolToSmiles(out_mol)


def standardize_mol_to_enhanced_smiles(
    mol: Optional[Chem.Mol],
    ops: List[str] = DEFAULT_OPS,
    policy: Policy = GPU_CPU_MANAGER,
    existing_meta: Optional[ChemAxonMeta] = None,
    index_base: int = 0,
    mode: str = "append",  # or "replace"
) -> Optional[str]:
    """
    Mol → standardized Mol → ChemAxon-style enhanced SMILES.
    """
    if mol is None:
        return None

    out_mol = standardize_mol(mol, ops=ops, policy=policy)
    if out_mol is None:
        return None

    return export_enhanced_smiles_from_mol(
        out_mol,
        existing_meta=existing_meta,
        index_base=index_base,
        mode=mode,
    )


def standardize_to_enhanced_smiles(
    smiles: str,
    ops: List[str] = DEFAULT_OPS,
    policy: Policy = GPU_CPU_MANAGER,
    index_base: int = 0,
    mode: str = "append",  # how to merge with original ChemAxon meta
) -> Optional[str]:
    """
    Enhanced SMILES (or plain SMILES) → standardized enhanced SMILES.
    """
    mol, meta = enhanced_smiles_to_mol(smiles, index_base=index_base)
    if mol is None:
        return None

    out_mol = standardize_mol(mol, ops=ops, policy=policy)
    if out_mol is None:
        return None

    return export_enhanced_smiles_from_mol(
        out_mol,
        existing_meta=meta,
        index_base=index_base,
        mode=mode,
    )
