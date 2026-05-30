"""evaluate.py - evaluation and demo utilities for nanoSASRec v4+"""

import math
import random
from typing import Dict, List, Tuple

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
) -> Tuple[float, float]:
    hrs, ndcgs = [], []
    eval_max_len = max_len - 1  # required to match training
    for user in range(1, num_users + 1):
        # generate user sequence based on mode
        user_seq = list(user_train[user])  # copy
        if mode == "test":
            # for test mode, add validation sequence to train
            user_seq = user_seq + user_valid[user]

        # pad to max sequence length
        seq_len = len(user_seq)
        if seq_len >= eval_max_len:
            # User sequences are temporally ordered from oldest to newest
            # For these models its best practice to keep the most recent interactions
            start_idx = seq_len - eval_max_len
            user_seq = user_seq[start_idx:]
        else:
            # 0 is reserved value for padding values
            num_pads = eval_max_len - seq_len
            pad_seq = [0] * num_pads
            user_seq = pad_seq + user_seq

        # convert to tensors and do model forward pass extracting last position as prediction
        input_seq = torch.tensor(user_seq, dtype=torch.long).expand(1, -1).to(device)
        output_seq = model.forward(input_seq)  # (batch_size, seq_len, embedding_dim)
        output_pred = output_seq[:, -1, :]

        # determine ground truth in leave one out evaluation methodology
        if mode == "valid":
            ground_truth = user_valid[user][0]
        else:
            ground_truth = user_test[user][0]

        # generate candidates to evaluate prediction against
        # sample items from unseen data
        if mode == "valid":
            seen = set(user_train[user]) | {ground_truth}
        else:
            seen = set(user_train[user]) | set(user_valid[user]) | {ground_truth}
        # draw 100 negative samples for eval (matches original SASRec protocol)
        # NOTE: sampled eval with 100 negatives inflates metrics vs full-catalog eval
        # full-catalog evaluation explored in later versions.
        negatives = []
        while len(negatives) < 100:
            item = random.randint(1, num_items)
            if item not in seen:
                negatives.append(item)
        # combine ground truth with sampled negatives
        # NOTE: ground truth must be first
        candidates = [ground_truth] + negatives

        # score candidates from learned embeddings against predicted position embedding
        candidate_ids = torch.tensor(candidates, dtype=torch.long).to(device)  # (101,)
        candidate_embeddings = model.item_embedding_table(candidate_ids)  # (101, embedding_dim)
        eval_scores = torch.multiply(output_pred, candidate_embeddings).sum(dim=-1)  # (101,)

        # rank of ground truth among all candidates (0-indexed)
        rank = (eval_scores > eval_scores[0]).sum().item()  # how many negatives scored higher

        # HR@10: did ground truth land in top 10?
        hr = 1.0 if rank < topk else 0.0
        hrs.append(hr)

        # NDCG@10: how high in the top 10?
        ndcg = (1.0 / math.log2(rank + 2)) if rank < topk else 0.0  # +2 because rank is 0-indexed
        ndcgs.append(ndcg)

    # average across all users to get HR@ and NDCG@ metrics
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
