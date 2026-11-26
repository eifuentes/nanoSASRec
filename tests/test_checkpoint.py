"""Tests that a resumed training run matches one that was never interrupted."""

import random
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from sasrec import checkpoint

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = ["train_v4.py", "train_v6.py", "train_v7.py", "train_v8.py", "train_v9.py", "train_v10.py", "train_v11.py"]


def test_resume_restores_every_rng_stream(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters())
    args = type("Args", (), {})()  # save() only records vars(args)
    torch.manual_seed(0), np.random.seed(0), random.seed(0)

    checkpoint.save(str(tmp_path), 1, model, optimizer, args)
    expected = (random.random(), np.random.rand(), torch.rand(1).item())

    # Draw more numbers so every generator moves on, then resume
    random.random(), np.random.rand(), torch.rand(1)
    assert checkpoint.resume(str(tmp_path), model, optimizer) == 1
    assert (random.random(), np.random.rand(), torch.rand(1).item()) == expected


def _train(script, data_dir, checkpoint_dir, epochs, eval_every, monkeypatch, extra=()):
    argv = [script, "--dataset", "ml-1m", "--data-dir", str(data_dir), "--device", "cpu"]
    argv += ["--dim", "16", "--blocks", "1", "--heads", "2", "--max-len", "8", "--batch-size", "4"]
    argv += ["--num-negatives", "4", "--epochs", str(epochs), "--eval-every", str(eval_every)]
    argv += ["--checkpoint-dir", str(checkpoint_dir), "--checkpoint-every", "1", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_path(str(REPO / script), run_name="__main__")
    (run,) = Path(checkpoint_dir).iterdir()  # one run directory per command
    return torch.load(run / f"epoch_{epochs:04d}.pt", weights_only=False)


@pytest.mark.parametrize("script", SCRIPTS)
def test_resumed_run_matches_uninterrupted_run(script, tiny_data_dir, tmp_path, monkeypatch, capsys):
    straight = _train(script, tiny_data_dir, tmp_path / "straight", 3, 3, monkeypatch)

    # Stop after 1 epoch, then run the same command with --epochs 3. It resumes from epoch 1.
    # A different --eval-every also checks that evaluation doesn't disturb the training RNGs.
    _train(script, tiny_data_dir, tmp_path / "resumed", 1, 1, monkeypatch)
    capsys.readouterr()
    resumed = _train(script, tiny_data_dir, tmp_path / "resumed", 3, 1, monkeypatch)
    # without this, a run that silently restarted from scratch would also match
    assert "(1 epochs completed)" in capsys.readouterr().out

    for name, weights in straight["model"].items():
        assert torch.equal(weights, resumed["model"][name]), name
    for param_id, state in straight["optimizer"]["state"].items():
        for key, value in state.items():
            assert torch.equal(value, resumed["optimizer"]["state"][param_id][key]), (param_id, key)


def test_deterministic_flag_resumes_the_same_run(tiny_data_dir, tmp_path, monkeypatch, capsys):
    # --deterministic only fixes the GPU's summation order, so a run started without it
    # can be resumed with it: same run directory, and on CPU the same weights.
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)  # undone after the test
    straight = _train("train_v7.py", tiny_data_dir, tmp_path / "straight", 2, 2, monkeypatch)
    _train("train_v7.py", tiny_data_dir, tmp_path / "resumed", 1, 1, monkeypatch)
    capsys.readouterr()
    try:
        resumed = _train("train_v7.py", tiny_data_dir, tmp_path / "resumed", 2, 1, monkeypatch, ["--deterministic"])
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(False)  # global setting: don't leak into other tests
    assert "(1 epochs completed)" in capsys.readouterr().out
    for name, weights in straight["model"].items():
        assert torch.equal(weights, resumed["model"][name]), name
