"""Tests for the model pieces every v4+ version is built from."""

import pytest
import torch

from sasrec.model import HSTU, SASRec
from sasrec.model.sasrec import bucket_relative_positions, bucket_temporal_distances

MAX_LEN = 9  # models see max_len - 1 = 8 positions
SEQ_LEN = MAX_LEN - 1
NUM_ITEMS = 30

MODELS = {
    "sasrec": lambda: SASRec(NUM_ITEMS, MAX_LEN, dropout_rate=0.0, embedding_dim=16, num_blocks=2, num_heads=2),
    "sasrec_rezero_relbias": lambda: SASRec(
        NUM_ITEMS,
        MAX_LEN,
        dropout_rate=0.0,
        embedding_dim=16,
        num_blocks=2,
        num_heads=2,
        scale_residual=True,
        relative_bias=True,
    ),
    "hstu": lambda: HSTU(NUM_ITEMS, MAX_LEN, dropout_rate=0.0, embedding_dim=16, num_blocks=2, num_heads=2),
}


def _inputs():
    torch.manual_seed(0)
    items = torch.randint(1, NUM_ITEMS + 1, (2, SEQ_LEN))
    timestamps = torch.cumsum(torch.randint(0, 100_000, (2, SEQ_LEN)), dim=1)
    return items, timestamps


@pytest.mark.parametrize("name", MODELS)
def test_output_has_one_vector_per_position(name):
    model = MODELS[name]().eval()
    items, timestamps = _inputs()
    with torch.no_grad():
        out = model(items, timestamps)
    assert out.shape == (2, SEQ_LEN, 16)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name", MODELS)
def test_model_cannot_see_the_future(name):
    # Causal masking: the output at position t may only depend on items 0..t.
    # Change the last item (and its timestamp) and every earlier output must stay the same.
    model = MODELS[name]().eval()
    # Causality is about the wiring, not the weights. Re-draw every weight so no branch starts
    # at zero: with ReZero-style init, attention adds nothing at first and the check would be vacuous.
    torch.manual_seed(1)
    for p in model.parameters():
        p.data.normal_(0, 0.5)
    items, timestamps = _inputs()
    changed_items, changed_timestamps = items.clone(), timestamps.clone()
    changed_items[:, -1] = items[:, -1] % NUM_ITEMS + 1  # a different valid item id
    changed_timestamps[:, -1] += 12_345
    with torch.no_grad():
        out = model(items, timestamps)
        changed = model(changed_items, changed_timestamps)
    assert torch.allclose(out[:, :-1], changed[:, :-1], atol=1e-6)
    assert not torch.allclose(out[:, -1], changed[:, -1])  # the last position does see it


def test_position_buckets_are_exact_up_to_15_then_log_spaced():
    distances = torch.arange(0, 199)
    buckets = bucket_relative_positions(distances)
    assert buckets[:16].tolist() == list(range(16))  # one bucket per distance for the 16 most recent
    assert buckets.max().item() == 31  # the farthest distance lands in the last bucket
    assert (buckets.diff() >= 0).all()  # farther never maps to a closer bucket
    assert bucket_relative_positions(torch.tensor([-3])).item() == 0  # future positions: masked anyway


def test_time_buckets_cover_seconds_to_30_days():
    day = 86_400
    seconds = torch.tensor([0, 1, 15, 60, 3_600, day, 30 * day, 365 * day])
    buckets = bucket_temporal_distances(seconds).tolist()
    assert buckets[:3] == [0, 1, 15]  # gaps under 16 seconds get their own bucket
    assert buckets == sorted(buckets)
    assert buckets[-2] == buckets[-1] == 31  # 30 days or more share the last bucket
