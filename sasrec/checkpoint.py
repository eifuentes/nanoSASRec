"""checkpoint.py - save and resume v4+ training runs.

Each run keeps its checkpoints in its own directory, named from the training
arguments that change what the run computes. Re-running the same command
finds that directory and continues from the latest checkpoint. This is what
happens when Modal preempts a container and restarts it with the same input.

A checkpoint holds the model, the optimizer, and the state of every random
number generator training draws from (Python for user sampling, NumPy for
negative sampling, torch for init and dropout). Restoring all of them makes a
resumed run produce the same weights as one that was never interrupted.
"""

import glob
import hashlib
import json
import os
import random
from argparse import Namespace

import numpy as np
import torch

# Arguments that don't change the weights a run produces, so they are left out of
# the run name. Leaving out --epochs means a finished run can be extended by
# re-running it with a larger --epochs (the learning rate is constant).
NOT_PART_OF_RUN = ("data_dir", "device", "epochs", "eval_every", "topk", "checkpoint_dir", "checkpoint_every")


def run_dir(checkpoint_dir: str, script: str, args: Namespace) -> str:
    """Directory for this run's checkpoints, e.g. checkpoints/train_v7-ml-20m-seed42-3f9a1c2e."""
    config = {k: v for k, v in vars(args).items() if k not in NOT_PART_OF_RUN}
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
    name = os.path.splitext(os.path.basename(script))[0]
    return os.path.join(checkpoint_dir, f"{name}-{args.dataset}-seed{args.seed}-{digest}")


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
    }


def _set_rng_state(state: dict):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if state["mps"] is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


def save(directory: str, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, args: Namespace):
    """Save the state after `epoch` completed epochs as epoch_XXXX.pt."""
    os.makedirs(directory, exist_ok=True)
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": _rng_state(),
        "args": vars(args),  # for humans inspecting the file; not used on restore
    }
    path = os.path.join(directory, f"epoch_{epoch:04d}.pt")
    # Write to a temporary file, then rename. A crash or preemption mid-write leaves
    # only a .tmp file, which resume() ignores, never a truncated checkpoint.
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)


def resume(directory: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    """Restore the latest checkpoint in `directory`, if any. Returns the number of completed epochs."""
    for path in sorted(glob.glob(os.path.join(directory, "epoch_*.pt")), reverse=True):
        try:
            # weights_only=False because the RNG states include NumPy arrays and tuples.
            # These are files this code wrote, not untrusted downloads.
            state = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:  # e.g. a file cut short by a crash outside the rename window
            print(f"WARNING: skipping unreadable checkpoint {path}: {e}")
            continue
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])  # moves optimizer state to the parameters' device
        _set_rng_state(state["rng"])
        print(f"Resumed from {path} ({state['epoch']} epochs completed)")
        return state["epoch"]
    print(f"No checkpoint in {directory}, starting from scratch")
    return 0
