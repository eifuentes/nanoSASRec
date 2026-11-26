"""Smoke tests: v1-v3 train and evaluate end to end on a tiny dataset.

v4+ get the same check, and more, from test_checkpoint.py.
"""

import math
import re
import runpy
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("script", ["train_v1.py", "train_v2.py", "train_v3.py"])
def test_script_trains_and_evaluates(script, small_data_dir, monkeypatch, capsys):
    argv = [script, "--dataset", "ml-1m", "--data-dir", str(small_data_dir), "--device", "cpu"]
    argv += ["--dim", "16", "--blocks", "1", "--heads", "2", "--max-len", "8", "--batch-size", "4"]
    argv += ["--epochs", "2", "--eval-every", "2"]
    if script != "train_v1.py":  # v1 always uses one negative
        argv += ["--num-negatives", "8"]
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_path(str(REPO / script), run_name="__main__")
    out = capsys.readouterr().out

    losses = [float(x) for x in re.findall(r"Avg loss: ([-\d.naif]+)", out)]
    assert len(losses) == 2 and all(math.isfinite(x) for x in losses)
    hr, ndcg = map(float, re.search(r"Test metrics: HR@10: ([\d.]+) NDCG@10: ([\d.]+)", out).groups())
    assert 0 <= ndcg <= hr <= 1
