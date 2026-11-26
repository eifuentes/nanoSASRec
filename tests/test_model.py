"""Tests for the model pieces every v4+ version is built from."""

import pytest
import torch

from sasrec.model import HSTU, SASRec
from sasrec.model.sasrec import AttentionBlock, bucket_relative_positions, bucket_temporal_distances

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
    "sasrec_original_block": lambda: SASRec(
        NUM_ITEMS, MAX_LEN, dropout_rate=0.0, embedding_dim=16, num_blocks=2, num_heads=2, block_style="original"
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
    buckets = bucket_relative_positions(distances, max_distance=198)
    assert buckets[:16].tolist() == list(range(16))  # one bucket per distance for the 16 most recent
    # each log bucket starts ~17% farther back than the one before (the numbers CURRICULUM quotes)
    starts = [int((buckets == b).nonzero()[0]) for b in range(16, 32)]
    assert starts == [16, 19, 22, 26, 31, 36, 42, 49, 57, 66, 78, 91, 106, 124, 145, 170]
    assert (buckets.diff() >= 0).all()  # farther never maps to a closer bucket
    assert bucket_relative_positions(torch.tensor([-3]), max_distance=198).item() == 0  # future: masked anyway


@pytest.mark.parametrize("model_class", [SASRec, HSTU])
def test_models_bucket_positions_once_up_front(model_class):
    # max_len 200, as in training: distances reach 198, so the log buckets are used too
    kwargs = {"relative_bias": True} if model_class is SASRec else {}
    model = model_class(NUM_ITEMS, 200, dropout_rate=0.0, embedding_dim=16, num_blocks=1, num_heads=2, **kwargs)
    positions = torch.arange(199)
    expected = bucket_relative_positions(positions.unsqueeze(1) - positions.unsqueeze(0), max_distance=198)
    assert torch.equal(model.position_buckets, expected)
    assert "position_buckets" not in model.state_dict()  # rebuilt on load, so checkpoints don't change


def test_time_buckets_cover_seconds_to_about_two_weeks():
    day = 86_400
    seconds = torch.tensor([0, 1, 15, 60, 3_600, day, 13 * day, 15 * day, 365 * day])
    buckets = bucket_temporal_distances(seconds).tolist()
    assert buckets[:3] == [0, 1, 15]  # gaps under 16 seconds get their own bucket
    assert buckets == sorted(buckets)
    # The cap is set to 30 days, but rounding down means the last bucket starts at ~14.2 days
    assert buckets[-3] == 30
    assert buckets[-2] == buckets[-1] == 31


@pytest.mark.parametrize("block_style", ["pre-ln", "original"])
def test_where_the_residual_starts(block_style):
    # ReZero scalars start at 0, which switches both sublayers off. What's left is the residual path:
    # pre-LN passes x through untouched, the original SASRec block re-normalizes it at every block.
    block = AttentionBlock(
        dropout_rate=0.0, embedding_dim=16, num_heads=2, scale_residual=True, block_style=block_style
    )
    x = 3 * torch.randn(2, SEQ_LEN, 16) + 1
    mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN), diagonal=1).bool()
    with torch.no_grad():
        out = block(x, mask)
    expected = x if block_style == "pre-ln" else block.norm_ffn(block.norm_attn(x))
    assert torch.allclose(out, expected, atol=1e-6)


def test_original_block_matches_the_sasrec_reference_code():
    # The SASRec authors' block (kang205/SASRec model.py; RecTools' SASRecTransformerLayer), written out
    torch.manual_seed(0)
    block = AttentionBlock(dropout_rate=0.0, embedding_dim=16, num_heads=2, block_style="original")
    x = torch.randn(2, SEQ_LEN, 16)
    mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN), diagonal=1).bool()
    with torch.no_grad():
        q = block.norm_attn(x)
        h = q + block.attn(q, mask, kv=x)  # queries normalized, keys and values not
        f = block.norm_ffn(h)
        expected = f + block.ffn(f)
        assert torch.allclose(block(x, mask), expected, atol=1e-6)
