import random

import numpy as np
import torch


def set_seed(seed: int):
    """Seed Python, NumPy, and Torch RNGs for reproducible training runs.

    This controls random user sampling, negative sampling via NumPy's global
    RNG, model initialization, and CUDA RNGs when available.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
