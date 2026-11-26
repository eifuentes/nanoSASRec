import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False):
    """Seed Python, NumPy, and Torch RNGs for reproducible training runs.

    This controls random user sampling, negative sampling via NumPy's global
    RNG, model initialization, and CUDA RNGs when available.

    Seeding alone makes CPU runs exact. On CUDA, a few operations sum their
    results in whatever order the GPU threads finish, so the last bits can
    change from run to run. In nanoSASRec that's the backward pass of the
    relative bias lookups (v7+): millions of positions read the same 32 rows,
    and their gradients are added up in a different order each time.
    `deterministic=True` makes PyTorch switch those operations to slower,
    fixed-order versions.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # cuBLAS needs this to give deterministic matrix multiplies. It must be set
        # before the first CUDA matrix multiply, so set it here, before training starts.
        # https://docs.pytorch.org/docs/stable/notes/randomness.html
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)


def get_git_sha() -> str:
    """Get current git SHA for experiment tracking."""
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"
