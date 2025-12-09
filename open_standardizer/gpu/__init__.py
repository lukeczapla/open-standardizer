"""
Auto-register GPU kernels if pybind11 / CUDA modules are importable.
"""

from ..gpu_ops import register_gpu_kernel

try:
    # These come from pybind11-wrapped CUDA objects
    from .standardize_gpu import (
        gpu_stereo,
        gpu_charge,
        gpu_bond_order,
        gpu_aromaticity,
        gpu_mesomerizer,
        gpu_keep_largest_fragment,
        gpu_final_clean,
        gpu_tautomerizer,
        gpu_remove_explicit_h,
        gpu_clear_isotopes
    )

    register_gpu_kernel("stereo", gpu_stereo)
    register_gpu_kernel("charge", gpu_charge)
    register_gpu_kernel("bond_order", gpu_bond_order)
    register_gpu_kernel("aromaticity", gpu_aromaticity)
    register_gpu_kernel("keep_largest_fragment", gpu_keep_largest_fragment)
    register_gpu_kernel("mesomerizer", gpu_mesomerizer)
    register_gpu_kernel("final_clean", gpu_final_clean)
    register_gpu_kernel("tautomerizer", gpu_tautomerizer)
    register_gpu_kernel("remove_explicit_h", gpu_remove_explicit_h)
    register_gpu_kernel("clear_isotopes", gpu_clear_isotopes)

except Exception as e:
    # No CUDA module installed — this is fine.
    print(f"[INFO] GPU kernels unavailable: {e}")
