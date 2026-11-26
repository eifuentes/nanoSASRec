"""Shared fixtures: a tiny dataset in the format data.py writes, so tests never download anything."""

import json
import random

import pytest


def write_split(root, num_items):
    """Write a 16-user split in the format data.py writes, with timestamps for v7+."""
    rng = random.Random(0)
    split = {key: {} for key in ("user_train", "user_valid", "user_test")}
    split.update({f"{key}_timestamps": {} for key in ("user_train", "user_valid", "user_test")})
    for user in range(1, 17):
        items = rng.sample(range(1, num_items + 1), k=rng.randint(10, 14))
        stamps = sorted(rng.sample(range(10_000), k=len(items)))
        for key, sl in (
            ("user_train", slice(None, -2)),
            ("user_valid", slice(-2, -1)),
            ("user_test", slice(-1, None)),
        ):
            split[key][str(user)] = items[sl]
            split[f"{key}_timestamps"][str(user)] = stamps[sl]
    split.update(num_users=16, num_items=num_items, item_id_map={str(i): str(i) for i in range(1, num_items + 1)})
    (root / "ml-1m").mkdir()
    (root / "ml-1m" / "split.json").write_text(json.dumps(split))
    return root


@pytest.fixture
def tiny_data_dir(tmp_path):
    """40 items: enough for v4+, which rank against the full catalog."""
    return write_split(tmp_path, num_items=40)


@pytest.fixture
def small_data_dir(tmp_path):
    """150 items: v1-v3 rank the ground truth against 100 sampled negatives, so they need more items."""
    return write_split(tmp_path, num_items=150)
