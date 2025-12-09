import socket
import shutil

def detect_cluster_environment() -> str:
    """
    Detect environment class:
      - "amp_gpu"      : inside AMP GPU node
      - "amp_cpu"      : inside AMP cluster but CPU node
      - "local_gpu"    : local workstation with CUDA
      - "local_cpu"    : default
    """

    host = socket.gethostname().lower()

    # AMP clusters often have recognizable hostname prefixes
    amp_prefixes = ["amp", "awsaivirl", "amp-studio", "compute", "gpu-"]
    if any(p in host for p in amp_prefixes):
        # Check if CUDA is available
        if shutil.which("nvidia-smi"):
            return "amp_gpu"
        return "amp_cpu"

    # Not AMP — detect local GPU
    if shutil.which("nvidia-smi"):
        return "local_gpu"

    return "local_cpu"


def get_policy_overrides(env: str):
    """
    Return strategy overrides depending on environment.
    Only overrides explicit fields — Policy() defaults fill the rest.

    Fields allowed:
      use_gpu: bool
      fallback_on_error: bool
      max_gpu_batch: int
      prefer_gpu_ops: list[str]
    """

    if env == "amp_gpu":
        return {
            "use_gpu": True,
            "fallback_on_error": True,
            "max_gpu_batch": 256,
            "prefer_gpu_ops": ["stereo", "aromaticity", "charge", "bond_order"],
        }

    if env == "amp_cpu":
        return {
            "use_gpu": False,
            "fallback_on_error": True,
        }

    if env == "local_gpu":
        return {
            "use_gpu": True,
            "fallback_on_error": True,
            "max_gpu_batch": 64,
        }

    # local_cpu
    return {
        "use_gpu": False,
        "fallback_on_error": True,
    }
