import time, os
from collections import deque
from rdkit import Chem
from typing import Mapping, Callable, Any

from .gpu_ops import GPU_OPS
from .cpu_ops import cpu_execute
from .gpu_exceptions import GPUNotAvailable, GPUStepFailed
from open_standardizer.cluster_env import detect_cluster_environment, get_policy_overrides
from .gpu_cpu_profiler import GPUCPUProfiler

USE_GPU = os.getenv("OPEN_STANDARDIZER_ENABLE_CUDA", "0") == "1"
if USE_GPU:
    from .gpu_ops import GPU_OPS
else:
    # CPU-only mode: no GPU kernels
    GPU_OPS: Mapping[str, Callable[..., Any]] = {}

class Policy:
    """
    Policy object passed to Standardizer, describes GPU/CPU usage choices
    and tracks failure rates, etc.
    """

    def __init__(
        self,
        enable_gpu: bool = True,
        max_gpu_fail_rate: float = 0.25,   # disable GPU if >25% failures in window
        window_size: int = 100,            # rolling failure window size
        allow_large_molecules: bool = False,
        max_atoms_gpu: int = 120,          # molecules above this skip GPU
        min_atoms_gpu: int = 2,            # very small ones skip GPU
        prefer_gpu: bool = True,
        gpu_required_ops: set | None = None,
        cpu_only_ops: set | None = None,
    ):
        self.enable_gpu = enable_gpu
        self.max_gpu_fail_rate = max_gpu_fail_rate
        self.window_size = window_size
        self.allow_large_molecules = allow_large_molecules
        self.max_atoms_gpu = max_atoms_gpu
        self.min_atoms_gpu = min_atoms_gpu
        self.prefer_gpu = prefer_gpu

        # operation groupings
        self.gpu_required_ops: set = gpu_required_ops or set()
        self.cpu_only_ops: set = cpu_only_ops or set()

        # GPU and CPU handlers
        self.gpu_ops = dict(GPU_OPS)
        self.cpu_ops = cpu_execute                    # CPU fallback shim

        # profiler
        self.profiler = GPUCPUProfiler(self)

        # internal failure tracking
        self._fail_window = deque(maxlen=window_size)
        self._gpu_disabled_until = 0.0

        # environment overrides
        env = detect_cluster_environment()
        overrides = get_policy_overrides(env)
        for k, v in overrides.items():
            setattr(self, k, v)

    # ------------------------------------------------------------------
    # Record GPU success/failure
    # ------------------------------------------------------------------
    def record_gpu_success(self) -> None:
        self._fail_window.append(0)

    def record_gpu_failure(self) -> None:
        self._fail_window.append(1)

    # ------------------------------------------------------------------
    # Is GPU temporarily disabled?
    # ------------------------------------------------------------------
    def gpu_temporarily_disabled(self) -> bool:
        return time.time() < self._gpu_disabled_until

    # ------------------------------------------------------------------
    # Evaluate failure rate
    # ------------------------------------------------------------------
    def gpu_failure_rate(self) -> float:
        if not self._fail_window:
            return 0.0
        return sum(self._fail_window) / len(self._fail_window)

    # ------------------------------------------------------------------
    # If GPU is failing often, disable for 30 seconds automatically
    # ------------------------------------------------------------------
    def check_disable_gpu(self) -> bool:
        rate = self.gpu_failure_rate()
        if rate > self.max_gpu_fail_rate:
            self._gpu_disabled_until = time.time() + 30.0
            return True
        return False

    # ------------------------------------------------------------------
    # Decide if a molecule should use GPU or CPU (molecule-level filter)
    # ------------------------------------------------------------------
    def should_use_gpu(self, mol: Chem.Mol) -> bool:
        if not self.enable_gpu:
            return False
        if self.gpu_temporarily_disabled():
            return False
        if not self.prefer_gpu:
            return False

        num_atoms = mol.GetNumAtoms()

        # Skip tiny molecules — GPU overhead is pointless
        if num_atoms < self.min_atoms_gpu:
            return False

        # Skip very large ones unless explicitly allowed
        if not self.allow_large_molecules and num_atoms > self.max_atoms_gpu:
            return False

        return True

    # ------------------------------------------------------------------
    # Resolve GPU function for op_name
    # ------------------------------------------------------------------
    def try_gpu(self, op_name: str):
        """
        Return the GPU handler for this operation, or None.
        """

        if not self.enable_gpu:
            return None

        # CPU-only ops skip GPU entirely
        if op_name in self.cpu_only_ops:
            return None

        # GPU-required op but no GPU function available → cannot use GPU
        if op_name in self.gpu_required_ops and op_name not in self.gpu_ops:
            return None

        # Let profiler weigh in (if it says CPU is better, skip GPU)
        best = self.profiler.best_for(op_name)
        if best != "gpu":
            return None

        return self.gpu_ops.get(op_name)


# ====================================================================================
# Public helper used by Standardizer and `standardize()`
# ====================================================================================

def try_gpu_standardize(smiles: str, pipeline, policy: Policy, gpu_fn):
    """
    Attempts GPU, updates policy metrics, returns either:
        - str output from GPU
        - None → meaning 'try CPU now'
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Molecule-based feasibility
    if not policy.should_use_gpu(mol):
        return None

    try:
        out = gpu_fn(smiles, pipeline)

        if isinstance(out, str) and out:
            policy.record_gpu_success()
            return out

        # GPU returned empty / None / bad type
        policy.record_gpu_failure()
        policy.check_disable_gpu()
        return None

    except (GPUNotAvailable, GPUStepFailed):
        # Expected GPU failures
        policy.record_gpu_failure()
        policy.check_disable_gpu()
        return None

    except Exception:
        # Any other failure → also count as GPU failure
        policy.record_gpu_failure()
        policy.check_disable_gpu()
        return None


# ====================================================================================
# GLOBAL INSTANCE (this is what `standardize.py` uses)
# ====================================================================================

GPU_CPU_MANAGER = Policy(
    enable_gpu=True,
    prefer_gpu=True,
    max_gpu_fail_rate=0.25,
    window_size=100,
    max_atoms_gpu=200,   # you already bumped this
    min_atoms_gpu=2,
    allow_large_molecules=True,
    gpu_required_ops={"clear_stereo", "aromatize", "charge_normalize",
                      "bond_order", "mesomerize"},
    cpu_only_ops={},
)
