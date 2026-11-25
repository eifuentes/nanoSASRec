"""Tests that a resumed training run matches one that was never interrupted."""

import json
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


@pytest.fixture
def tiny_data_dir(tmp_path):
    """A 16-user, 40-item split in the format data.py writes, with timestamps for v7+."""
    rng = random.Random(0)
    split = {key: {} for key in ("user_train", "user_valid", "user_test")}
    split.update({f"{key}_timestamps": {} for key in ("user_train", "user_valid", "user_test")})
    for user in range(1, 17):
        items = rng.sample(range(1, 41), k=rng.randint(10, 14))
        stamps = sorted(rng.sample(range(10_000), k=len(items)))
        for key, sl in (
            ("user_train", slice(None, -2)),
            ("user_valid", slice(-2, -1)),
            ("user_test", slice(-1, None)),
        ):
            split[key][str(user)] = items[sl]
            split[f"{key}_timestamps"][str(user)] = stamps[sl]
    split.update(num_users=16, num_items=40, item_id_map={str(i): str(i) for i in range(1, 41)})
    (tmp_path / "ml-1m").mkdir()
    (tmp_path / "ml-1m" / "split.json").write_text(json.dumps(split))
    return tmp_path


def _train(script, data_dir, checkpoint_dir, epochs, eval_every, monkeypatch):
    argv = [script, "--dataset", "ml-1m", "--data-dir", str(data_dir), "--device", "cpu"]
    argv += ["--dim", "16", "--blocks", "1", "--heads", "2", "--max-len", "8", "--batch-size", "4"]
    argv += ["--num-negatives", "4", "--epochs", str(epochs), "--eval-every", str(eval_every)]
    argv += ["--checkpoint-dir", str(checkpoint_dir), "--checkpoint-every", "1"]
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
