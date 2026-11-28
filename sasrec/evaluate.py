"""evaluate.py - evaluation and demo utilities for nanoSASRec v6+ (train_v4.py has its own copy)"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch

from .model.hstu import HSTU
from .model.sasrec import SASRec


def _score_full_catalog(
    pred: torch.Tensor,
    all_item_embs: torch.Tensor,
    seen: set,
    device: torch.device,
) -> torch.Tensor:
    """Score a single user's prediction against all items in the catalog.

    Pure scoring logic: matmul + seen masking. Any embedding normalization
    (e.g., L2 norm for cosine similarity) should be applied by the caller
    before passing tensors here.

    Args:
        pred: (embedding_dim,) the user's predicted embedding
        all_item_embs: (num_items, embedding_dim) pre-computed item embeddings
        seen: set of item IDs to mask (already interacted)
        device: torch device

    Returns:
        scores: (num_items,) score per item, seen items set to -inf
    """
    scores = torch.matmul(pred, all_item_embs.T)  # (num_items,)
    # mask seen items using tensor indexing (avoids slow Python loop on GPU)
    seen_indices = torch.tensor(list(seen), dtype=torch.long, device=device) - 1
    scores[seen_indices] = float("-inf")
    return scores


def evaluate(
    model: Union[SASRec, HSTU],
    user_train: Dict[int, List[int]],
    user_valid: Dict[int, List[int]],
    user_test: Dict[int, List[int]],
    num_users: int,
    num_items: int,
    max_len: int,
    device: torch.device,
    mode: str = "valid",
    topk: int = 10,
    eval_batch_size: int = 256,
    user_train_ts: Optional[Dict[int, List[int]]] = None,
    user_valid_ts: Optional[Dict[int, List[int]]] = None,
    l2_norm: bool = False,
) -> Tuple[float, float]:
    """Full-catalog HR@K and NDCG@K on the validation or test split.

    mode="valid" ranks the second-to-last item given the train history.
    mode="test" ranks the last item given the train + valid history.

    The ground truth is ranked against every item in the catalog except the ones
    already in the user's input history. Both splits use the full catalog: metrics
    from 100 sampled negatives are inflated and can change which model looks better
    (Krichene & Rendle, KDD 2020). Validation and test numbers are therefore on the
    same scale.
    """
    hrs, ndcgs = [], []
    eval_max_len = max_len - 1

    # Pre-build all user sequences, timestamps, and ground truths
    all_seqs = []
    all_ts = []
    all_gts = []
    all_seens = []
    has_timestamps = user_train_ts is not None and len(user_train_ts) > 0

    for user in range(1, num_users + 1):
        user_seq = list(user_train[user])
        if mode == "test":
            user_seq = user_seq + user_valid[user]

        seq_len = len(user_seq)
        if seq_len >= eval_max_len:
            user_seq = user_seq[seq_len - eval_max_len :]
        else:
            user_seq = [0] * (eval_max_len - seq_len) + user_seq
        all_seqs.append(user_seq)

        # Build parallel timestamp sequence
        if has_timestamps and user_train_ts is not None:
            ts_seq = list(user_train_ts[user])
            if mode == "test" and user_valid_ts:
                ts_seq = ts_seq + user_valid_ts[user]
            ts_len = len(ts_seq)
            if ts_len >= eval_max_len:
                ts_seq = ts_seq[ts_len - eval_max_len :]
            else:
                ts_seq = [0] * (eval_max_len - ts_len) + ts_seq
            all_ts.append(ts_seq)

        # seen = every item the user interacted with before the ground truth, not only
        # the last max_len the model reads. These are masked out of the ranking, as in
        # eSASRec (RecTools' filter_viewed). This assumes no repeat consumption, which
        # holds for MovieLens: a repeated ground truth would be masked and count as a miss.
        if mode == "valid":
            all_gts.append(user_valid[user][0])
            all_seens.append(set(user_train[user]))
        else:
            all_gts.append(user_test[user][0])
            all_seens.append(set(user_train[user]) | set(user_valid[user]))

    # Pre-compute all item embeddings once, shared by every batch. Item ids start at 1
    # because 0 is padding, so row i holds item i + 1 and item id - 1 is its column.
    all_item_embs = model.item_embedding_table(
        torch.arange(1, num_items + 1, device=device)
    )  # (num_items, embedding_dim)
    # L2 normalize item embeddings upstream so all downstream scoring uses cosine similarity
    if l2_norm:
        all_item_embs = torch.nn.functional.normalize(all_item_embs, p=2, dim=-1, eps=1e-6)

    # Process in batches
    for start in range(0, num_users, eval_batch_size):
        end = min(start + eval_batch_size, num_users)
        batch_seqs = all_seqs[start:end]
        batch_gts = all_gts[start:end]
        batch_seens = all_seens[start:end]

        # Batched forward pass
        input_batch = torch.tensor(batch_seqs, dtype=torch.long, device=device)
        if has_timestamps:
            batch_ts = all_ts[start:end]
            ts_batch = torch.tensor(batch_ts, dtype=torch.long, device=device)
            output_batch = model(input_batch, timestamps=ts_batch)
        else:
            output_batch = model(input_batch)
        pred_batch = output_batch[:, -1, :]  # (batch_size, embedding_dim)

        # L2 normalize output embeddings upstream so all downstream scoring uses cosine similarity
        if l2_norm:
            pred_batch = torch.nn.functional.normalize(pred_batch, p=2, dim=-1, eps=1e-6)

        # Score every item for every user in the batch with one matmul
        scores_batch = torch.matmul(pred_batch, all_item_embs.T)  # (batch_size, num_items)

        # Mask seen items with one indexed assignment for the whole batch, instead of
        # one small tensor op per user. Each (row, col) pair is (user's row in the batch,
        # seen item's column).
        seen_rows, seen_cols = [], []
        for row, seen in enumerate(batch_seens):
            for item in seen:
                seen_rows.append(row)
                seen_cols.append(item - 1)
        scores_batch[
            torch.tensor(seen_rows, dtype=torch.long, device=device),
            torch.tensor(seen_cols, dtype=torch.long, device=device),
        ] = float("-inf")

        # Rank of the ground truth = how many items score strictly higher than it
        gt_indices = torch.tensor(batch_gts, dtype=torch.long, device=device) - 1  # item id -> column
        gt_scores = scores_batch[torch.arange(end - start, device=device), gt_indices]  # (batch_size,)
        ranks = (scores_batch > gt_scores.unsqueeze(1)).sum(dim=1)  # (batch_size,)

        # Convert ranks to metrics
        for rank in ranks.cpu().tolist():
            hrs.append(1.0 if rank < topk else 0.0)
            ndcgs.append((1.0 / math.log2(rank + 2)) if rank < topk else 0.0)

    hr_at = sum(hrs) / num_users
    ndcg_at = sum(ndcgs) / num_users
    return hr_at, ndcg_at


def demo(
    user,
    model: Union[SASRec, HSTU],
    user_train: Dict[int, List[int]],
    user_valid: Dict[int, List[int]],
    user_test: Dict[int, List[int]],
    num_items: int,
    max_len: int,
    device: torch.device,
    titles: Dict[int, str],
    user_train_ts: Optional[Dict[int, List[int]]] = None,
    user_valid_ts: Optional[Dict[int, List[int]]] = None,
    l2_norm: bool = False,
):
    def name(item_id):
        return titles.get(item_id, f"item_{item_id}")

    # build input sequence (train + valid, same as test eval)
    user_seq = list(user_train[user]) + user_valid[user]
    eval_max_len = max_len - 1
    if len(user_seq) >= eval_max_len:
        user_seq = user_seq[-eval_max_len:]
    else:
        user_seq = [0] * (eval_max_len - len(user_seq)) + user_seq
    ground_truth = user_test[user][0]

    # build timestamp sequence if available
    has_timestamps = user_train_ts is not None and len(user_train_ts) > 0
    ts_seq = None
    if has_timestamps and user_train_ts is not None and user_valid_ts is not None:
        ts_seq = list(user_train_ts[user]) + user_valid_ts[user]
        if len(ts_seq) >= eval_max_len:
            ts_seq = ts_seq[-eval_max_len:]
        else:
            ts_seq = [0] * (eval_max_len - len(ts_seq)) + ts_seq

    # forward pass
    input_seq = torch.tensor(user_seq, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        if has_timestamps:
            ts_tensor = torch.tensor(ts_seq, dtype=torch.long, device=device).unsqueeze(0)
            output = model(input_seq, timestamps=ts_tensor)
        else:
            output = model(input_seq)
        pred = output[:, -1, :]  # (1, embedding_dim)

    # full-catalog scoring using shared helper
    all_items = torch.arange(1, num_items + 1, device=device)
    all_item_embs = model.item_embedding_table(all_items)

    # L2 normalize upstream so _score_full_catalog uses cosine similarity
    if l2_norm:
        pred = torch.nn.functional.normalize(pred, p=2, dim=-1, eps=1e-6)
        all_item_embs = torch.nn.functional.normalize(all_item_embs, p=2, dim=-1, eps=1e-6)

    seen = set(user_train[user]) | set(user_valid[user])
    scores = _score_full_catalog(pred.squeeze(0), all_item_embs, seen, device)

    # top 10
    top10 = (scores.topk(10).indices + 1).cpu().tolist()

    print(f"\n  Prediction demo (user {user}):")
    print(f"    Recent history:  {', '.join(name(i) for i in user_train[user][-5:])}")
    print(f"    Ground truth:    {name(ground_truth)}")
    print("    Model top-10:")
    for rank, item_id in enumerate(top10, 1):
        marker = " <- hit" if item_id == ground_truth else ""
        print(f"      {rank:2d}. {name(item_id)}{marker}")
    print()
