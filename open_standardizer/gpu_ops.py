# molvs_standardizer/gpu_ops.py

import os
from typing import Callable, List, Dict

USE_GPU = os.getenv("OPEN_STANDARDIZER_ENABLE_CUDA", "0").lower() in ["1", "true", "yes"]

if USE_GPU:
    from .gpu.standardize_gpu import gpu_standardize


GPU_OPS: Dict[str, Callable[[str, List[str]], str]] = {}

def _single_step(step_name: str):
    def _fn(smiles: str, pipeline: List[str]) -> str:
        # We ignore `pipeline` here and just run the requested step.
        # The step_name should match the ones registered in standardize_gpu.
        return gpu_standardize(smiles, steps=[step_name])
    return _fn

GPU_OPS.update(
    {
        "clear_isotopes":              _single_step("clear_isotopes"),   # if you register it
        "clear_stereo":                _single_step("stereo"),
        "bond_order":                  _single_step("bond_order"),
        "charge":                      _single_step("charge_normalize"),
        "remove_largest_fragment":     _single_step("keep_largest_fragment"),
        "remove_fragment_keeplargest": _single_step("keep_largest_fragment"),
        "remove_explicit_h":           _single_step("remove_explicit_h"),
        "aromatize":                   _single_step("aromaticity"),
        "mesomerize":                  _single_step("mesomerize"),
        "tautomerize":                 _single_step("tautomerize"),
    }
)


def _make_single_step_gpu(step_name: str) -> Callable[[str, List[str]], str]:
    """
    Wrap gpu_standardize() so it looks like a per-op GPU function:

        fn(smiles: str, pipeline: list[str]) -> str

    The 'pipeline' arg is ignored here; just run the single GPU step.
    """
    def _fn(smiles: str, pipeline: List[str]) -> str:
        return gpu_standardize(smiles, steps=[step_name])
    return _fn


def register_gpu_ops(policy) -> None:
    """
    Populate policy.gpu_ops with per-operation GPU handlers.
    Op names here must match whatever your CPU ops / XML mapping uses.
    """

    # Map high-level op names → GPU step names
    # Adjust keys to match your cpu_ops naming!
    policy.gpu_ops.update({
        # ChemAxon "ClearStereo" equivalent
        "clear_stereo": _make_single_step_gpu("stereo"),

        # Charge normalization
        "charge_normalize": _make_single_step_gpu("charge"),

        # Bond-order inference
        "bond_order": _make_single_step_gpu("bond_order"),

        # Aromaticity perception
        "aromatize": _make_single_step_gpu("aromaticity"),

        # Mesomerize / resonance normalization
        "mesomerize": _make_single_step_gpu("mesomerizer"),

        "tautomerize": _make_single_step_gpu("tautomerizer"),
    })

    # Optionally: mark which ops are GPU-friendly / CPU-only
    if hasattr(policy, "gpu_required_ops"):
        policy.gpu_required_ops.update({
            "clear_stereo",
            "aromatize",
            "mesomerize",
            "charge_normalize",
            "bond_order",
            "tautomerize"
        })

    if hasattr(policy, "cpu_only_ops"):
        policy.cpu_only_ops.update({
            # If there are ops you explicitly never want on GPU:
        })


