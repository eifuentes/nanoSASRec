"""evaluate.py - evaluation and demo utilities for nanoSASRec v4+"""

import math
import random
from typing import Dict, List, Optional, Tuple

import torch
from .model import SASRec

def evaluate(
    model: SASRec,
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
) -> Tuple[float, float]:
    hrs, ndcgs = [], []
    eval_max_len = max_len - 1

    # Pre-build all user sequences, timestamps, and ground truths
    all_seqs = []
    all_ts = []
    all_gts = []
    all_seens = []
    # When timestamps are available, pass them to the model so temporal bias
    # is active during evaluation (matching training conditions).
    # Without this, eval only uses positional bias, underestimating the model.
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

        if mode == "valid":
            all_gts.append(user_valid[user][0])
            all_seens.append(set(user_train[user]) | {user_valid[user][0]})
        else:
            all_gts.append(user_test[user][0])
            all_seens.append(set(user_train[user]) | set(user_valid[user]) | {user_test[user][0]})

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
        pred_batch = output_batch[:, -1, :]  # (batch, embedding_dim)

        # Score candidates per user (negatives differ per user)
        for i in range(end - start):
            ground_truth = batch_gts[i]
            seen = batch_seens[i]

            negatives = []
            while len(negatives) < 100:
                item = random.randint(1, num_items)
                if item not in seen:
                    negatives.append(item)
            candidates = [ground_truth] + negatives

            candidate_ids = torch.tensor(candidates, dtype=torch.long, device=device)
            candidate_embs = model.item_embedding_table(candidate_ids)
            scores = torch.multiply(pred_batch[i], candidate_embs).sum(dim=-1)

            rank = (scores > scores[0]).sum().item()
            hrs.append(1.0 if rank < topk else 0.0)
            ndcgs.append((1.0 / math.log2(rank + 2)) if rank < topk else 0.0)

    hr_at = sum(hrs) / num_users
    ndcg_at = sum(ndcgs) / num_users
    return hr_at, ndcg_at


def demo(user, model, user_train, user_valid, user_test, num_items, max_len, device, titles):
    def name(item_id):
        return titles.get(item_id, f"item_{item_id}")

    # build input sequence (train + valid, same as test eval)
    user_seq = list(user_train[user]) + user_valid[user]
    max_len = max_len - 1
    if len(user_seq) >= max_len:
        user_seq = user_seq[-max_len:]
    else:
        user_seq = [0] * (max_len - len(user_seq)) + user_seq
    ground_truth = user_test[user][0]

    # score all items in catalog
    input_seq = torch.tensor(user_seq, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        output = model(input_seq)
        pred = output[:, -1, :]  # last position (1, embedding_dim)
        all_items = torch.arange(1, num_items + 1, device=device)
        all_emb = model.item_embedding_table(all_items)  # (num_items, embedding_dim)
        scores = (pred * all_emb).sum(dim=-1)  # (num_items,)

    # mask out seen items
    seen = set(user_train[user]) | set(user_valid[user])
    for item in seen:
        scores[item - 1] = float("-inf")  # -1 because items are 1-indexed, scores are 0-indexed

    # top 10
    top10 = (scores.topk(10).indices + 1).cpu().tolist()  # back to 1-indexed

    print(f"\n  Prediction demo (user {user}):")
    print(f"    Recent history:  {', '.join(name(i) for i in user_train[user][-5:])}")
    print(f"    Ground truth:    {name(ground_truth)}")
    print("    Model top-10:")
    for rank, item_id in enumerate(top10, 1):
        marker = " <- hit" if item_id == ground_truth else ""
        print(f"      {rank:2d}. {name(item_id)}{marker}")
    print()
