"""Tests for how v6+ build a training batch. train_v4.py has the same sampler without timestamps."""

import numpy as np

from sasrec.dataset import sample_batch


def test_batch_shifts_targets_pads_left_and_never_samples_seen_items():
    np.random.seed(0)
    user_train = {1: [3, 5, 7], 2: [1, 2, 4, 6, 8, 9, 10]}
    timestamps = {1: [10, 20, 30], 2: [1, 2, 3, 4, 5, 6, 7]}
    max_len = 5  # keeps the 5 most recent items: 4 inputs and 4 targets

    inputs, targets, negatives, ts = sample_batch(
        user_train, num_users=2, num_items=12, max_len=max_len, batch_size=2, num_negs=4, timestamps=timestamps
    )
    rows = {int(row[-1]): i for i, row in enumerate(targets)}  # last target identifies the user
    u1, u2 = rows[7], rows[10]

    # A short history is padded with 0 on the left, and the target is the input shifted by one.
    assert inputs[u1].tolist() == [0, 0, 3, 5]
    assert targets[u1].tolist() == [0, 3, 5, 7]
    assert ts[u1].tolist() == [0, 0, 10, 20]
    # A long history keeps only its most recent items.
    assert inputs[u2].tolist() == [4, 6, 8, 9]
    assert targets[u2].tolist() == [6, 8, 9, 10]

    # Negatives skip everything in the user's history, so a negative can never equal a target.
    # This is why v10's collision masking never fires with this sampler.
    for row, user in ((u1, 1), (u2, 2)):
        assert not set(negatives[row].tolist()) & set(user_train[user])
        assert all(1 <= n <= 12 for n in negatives[row])  # 0 is padding, never a negative
