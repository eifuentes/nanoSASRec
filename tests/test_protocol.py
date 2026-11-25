"""Tests for the evaluation-protocol decisions that published comparisons depend on."""

import math

import torch

from data import reindex_and_split
from sasrec.evaluate import evaluate


def test_split_sorts_by_time_and_keeps_file_order_on_ties():
    # u1's last two interactions share a timestamp. File order (d, then c) must decide
    # which is valid and which is test, not the item id. Users are interleaved in the
    # file, and u2 has too few interactions to split.
    interactions = [
        ("u1", "b", 2),
        ("u2", "x", 5),
        ("u1", "a", 1),
        ("u1", "d", 3),
        ("u2", "y", 6),
        ("u1", "c", 3),
    ]
    split = reindex_and_split(interactions)

    # ids start at 1 (0 is padding) and follow sorted raw ids: a=1 b=2 c=3 d=4 x=5 y=6
    assert split.num_users == 2
    assert split.num_items == 6
    assert split.user_train[1] == [1, 2]
    assert split.user_valid[1] == [4]  # d
    assert split.user_test[1] == [3]  # c
    assert split.user_train_ts[1] == [1, 2]
    assert split.user_valid_ts[1] == [3]
    assert split.user_test_ts[1] == [3]
    assert split.user_train[2] == [5, 6]
    assert split.user_valid[2] == []
    assert split.user_test[2] == []


class FixedScoreModel(torch.nn.Module):
    """A 1-d "model" whose score for item i is exactly item_scores[i - 1]."""

    def __init__(self, item_scores):
        super().__init__()
        weights = torch.tensor([[0.0]] + [[s] for s in item_scores])  # row 0 = padding
        self.item_embedding_table = torch.nn.Embedding.from_pretrained(weights)

    def forward(self, input_seqs):
        return torch.ones(input_seqs.shape[0], input_seqs.shape[1], 1)


def test_evaluate_ranks_against_full_catalog_and_masks_history():
    # One user: train = [1, 2], valid = [3], test = [4]. Items 1-3 outscore everything,
    # so unless the history is masked they push the ground truth down the ranking.
    # Items 5 and 6 are the only unseen items that beat the test item.
    scores = [100, 99, 90, 50, 60, 55, 1, 2, 3, 4, 5, 6]
    model = FixedScoreModel(scores)
    split = dict(user_train={1: [1, 2]}, user_valid={1: [3]}, user_test={1: [4]}, num_users=1)
    kwargs = dict(num_items=len(scores), max_len=5, device=torch.device("cpu"), topk=3)

    with torch.no_grad():
        # valid: train items masked, so item 3 ranks first
        assert evaluate(model, **split, **kwargs, mode="valid") == (1.0, 1.0)
        # test: train and valid items masked, so items 5 and 6 are the only ones above item 4
        hr, ndcg = evaluate(model, **split, **kwargs, mode="test")
        assert hr == 1.0
        assert math.isclose(ndcg, 1 / math.log2(2 + 2))
        # same ranking, but rank 2 falls outside the top 2
        assert evaluate(model, **split, **{**kwargs, "topk": 2}, mode="test") == (0.0, 0.0)
