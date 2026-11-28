"""
train_v4.py - SASRec architecture refinements + Sampled Softmax reintroduced

Single-file implementation, like v1-v3, and the last one: v5 runs this script on ML-20M,
and v6+ import the shared sasrec/ package instead.

Implementation refinements: truncated normal weight init, reversed positional
embeddings, registered causal mask buffer, configurable feed-forward network, attention dropout.
Also new in v4: full-catalog evaluation, resumable checkpoints, and two flags that rewire
each attention block (--attention-block-residual-from, --attention-block-kv-from) to
compare pre-LN with the SASRec authors' released code (CURRICULUM v4 and v5, issue #32).
Loss function unchanged from v2 (Sampled Softmax), reintroduced.

Run using the command: `uv run python train_v4.py --epochs 200 --eval-every 20`
Be sure to specify the device you want to run on using the --device argument.

References:
    - Kang & McAuley, 2018 (arxiv: 1808.09781) - SASRec architecture
    - Huang, 2025 (arxiv: 2504.09596) - positional embedding alignment
    - D'Ercoli et al., 2025 (arxiv: 2506.14692) - implementation details that move metrics
    - Klenitskiy & Vasilev, RecSys 2023 (arxiv: 2309.07602) - Sampled Softmax loss
    - Krichene & Rendle, KDD 2020 - sampled evaluation metrics are inflated
"""

import argparse
import glob
import hashlib
import json
import math
import os
import random
import subprocess
from argparse import Namespace
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

# Possible datasets for training
DATASETS = ("ml-1m", "ml-20m")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_sha() -> str:
    """Get current git SHA for experiment tracking."""
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Data loading and batch sampling
# ---------------------------------------------------------------------------


def load_data(
    data_dir: str, dataset: str
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, List[int]], int, int, Dict[int, str]]:
    """Load preprocessed dataset split from disk.

    Returns: user_train, user_valid, user_test, num_users, num_items, item_id_map
    """
    dataset_split_filepath = f"{data_dir}/{dataset}/split.json"
    assert os.path.exists(dataset_split_filepath), f"Run `python data.py --dataset {dataset}` first"
    with open(dataset_split_filepath) as f:
        dataset_split = json.load(f)

    user_train = {int(k): v for k, v in dataset_split["user_train"].items()}
    user_valid = {int(k): v for k, v in dataset_split["user_valid"].items()}
    user_test = {int(k): v for k, v in dataset_split["user_test"].items()}

    num_users: int = dataset_split["num_users"]
    num_items: int = dataset_split["num_items"]

    item_id_map = {int(k): v for k, v in dataset_split["item_id_map"].items()}

    print(f"Number Users: {num_users:,}\nNumber Items: {num_items:,}")
    return user_train, user_valid, user_test, num_users, num_items, item_id_map


def load_item_titles(data_dir: str, dataset: str, item_id_map: Dict[int, str]) -> Dict[int, str]:
    titles_path = f"{data_dir}/{dataset}/titles.json"
    if not os.path.exists(titles_path):
        return {}
    with open(titles_path) as f:
        raw_titles = json.load(f)
    return {idx: raw_titles[raw_id] for idx, raw_id in item_id_map.items() if raw_id in raw_titles}


def sample_batch(
    user_train: Dict[int, List[int]], num_users: int, num_items: int, max_len: int, batch_size: int, num_negs: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs a training batch for autoregressive sequence recommendation models like SASRec.

    Returns three numpy arrays each comprised of sampled users of batch_size to create the batch
        - input of user's sequences truncated to remove the last item, shape(batch_size, max_len -1)
        - positive targets are the shifted-by-one ground truth next items, shape(batch_size, max_len -1)
        - target of per user unseen items treated as negatives to be used in contrastive training, shape(batch_size, num_negs)

    We use NumPy arrays for the batches because they are
    memory-contiguous, which is required for efficient GPU processing.
    PyTorch later handles these NumPy arrays very gracefully.
    """
    # 1. Sample users
    # NOTE: random.sample is 'without replacement', gives unique users
    batch_users = random.sample(range(1, num_users + 1), k=batch_size)

    # 2. Pre-allocate matrices in memory of type int64 which is equal to torch LongTensor later
    # NOTE: (max_len - 1) is a result of the position prediction setup of autoregressive models like SASRec
    input_seq_batch = np.zeros((batch_size, max_len - 1), dtype=np.int64)
    target_pos_seq_batch = np.zeros((batch_size, max_len - 1), dtype=np.int64)
    # NOTE: We sample a set of negatives per user that get reused for each position/target combination
    target_neg_seq_batch = np.zeros((batch_size, num_negs), dtype=np.int64)

    # each record in the batch is a user sampled
    for i, user_id in enumerate(batch_users):
        user_seq: List[int] = user_train[user_id]

        # Convert to set once to be more performant
        # in python 'item in list' is O(N) -> slow. 'item in set' is O(1) -> instant.
        # we check against the full user history before truncation so negatives are truly "unseen" items.
        seen_items = set(user_seq)

        # 3. Handle sequence padding or truncation depending on length of sequence
        user_seq_len = len(user_seq)
        if user_seq_len >= max_len:
            # Keep only the 'max_len' most recent interactions
            # NOTE: the sequence is ordered by time starting with oldest to newest
            user_seq = user_seq[user_seq_len - max_len :]
        else:
            # Pad with 0 at the beginning (pre-padding)
            user_seq = [0] * (max_len - user_seq_len) + user_seq

        # 4. Shift sequences for autoregressive prediction
        # input is 0 to t-1, target is 1 to t
        input_user_seq = user_seq[:-1]
        target_user_seq = user_seq[1:]

        # 5. Sample `num_negs` negatives per user using a rejection sampling approach
        user_negs = np.zeros(num_negs, dtype=np.int64)

        # Rejection sampling loop
        count = 0
        while count < num_negs:
            # Sample a block of random IDs in C-speed, then filter in python.
            # Uses NumPy's global RNG so set_seed(...) controls sampling.
            # NOTE: `num_items + 1` is due to exclusive nature of randint on the high side so all item ids are included
            # NOTE: we start with 1 in randint since 0 is reserved for the padding token in the item ID lookups
            samples = np.random.randint(1, num_items + 1, size=num_negs - count)
            for s in samples:
                if s not in seen_items:
                    user_negs[count] = s
                    count += 1
                    if count == num_negs:
                        break

        # 6. Move local sequences into the batch matrices
        input_seq_batch[i, :] = input_user_seq
        target_pos_seq_batch[i, :] = target_user_seq
        target_neg_seq_batch[i, :] = user_negs

    return input_seq_batch, target_pos_seq_batch, target_neg_seq_batch


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def init_weights(module: nn.Module, embedding_dim: int):
    """Initialize weights using truncated normal distribution.

    PyTorch defaults (Kaiming uniform for Linear, N(0,1) for nn.Embedding)
    produce dot products with variance ~embedding_dim, saturating softmax
    and causing vanishing gradients in early training. Truncated normal
    with std=1/sqrt(d) keeps initial scores near zero for stable training.
    """
    std = 1.0 / (embedding_dim**0.5)
    if isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, dropout_rate: float, num_heads: int = 1):

        super().__init__()

        # args
        self.num_heads = num_heads

        # setup the attention projections
        self.query_proj = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False)
        self.key_proj = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False)
        self.value_proj = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False)
        self.out_proj = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False)
        # add attention dropout
        self.attn_dropout = nn.Dropout(p=dropout_rate)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, key_value_input: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Scaled Dot-Product Self-Attention.

        Args:
            x (torch.Tensor): input for the queries, of size(batch_size, seq_len, embedding_dim)
            mask (torch.Tensor): mask to preserve causal nature of attention scores, of size(seq_len, seq_len)
                or larger (it is sliced to seq_len)
            key_value_input (torch.Tensor, optional): input for the keys and values, of
                size(batch_size, seq_len, embedding_dim). Defaults to x, so queries, keys and values
                all read the same tensor. AttentionBlock passes a different tensor only for
                --attention-block-kv-from input, where the queries read LayerNorm(x) and the keys
                and values read x.

        Returns:
            torch.Tensor: attended input tensor of size(batch_size, seq_len, embedding_dim)
        """
        # input tensor x is of shape (batch_size, seq_len, embedding_dim)
        batch_size, seq_len, embedding_dim = x.size()
        # keys and values read the same tensor as the queries unless told otherwise
        key_value_input = x if key_value_input is None else key_value_input

        # infer attention head dimensions from argument num_heads, should be cleanly divisible
        head_dim = embedding_dim // self.num_heads

        # project query, key, value and calculate attention scores
        # each linear layer contains all weights for all heads for efficiency and later breaks them out via reshaping

        # query linear layer transforms each position into "this is what I am looking for?"
        query = self.query_proj(x)  # (batch_size, seq_len, embedding_dim)
        query = query.view(batch_size, seq_len, self.num_heads, head_dim).transpose(
            1, 2
        )  # (batch_size, num_heads, seq_len, head_dim)

        # key linear layer transforms each position into "this is what I advertise about myself so queries can find me?"
        key = self.key_proj(key_value_input)  # (batch_size, seq_len, embedding_dim)
        key = key.view(batch_size, seq_len, self.num_heads, head_dim).transpose(
            1, 2
        )  # (batch_size, num_heads, seq_len, head_dim)
        # transpose key for future matrix multiply so dimensions line up with query tensor
        key = key.transpose(-2, -1)  # (batch_size, num_heads, head_dim, seq_len)

        # value linear layer transforms each position into "this is the information I pass along when attended to?"
        value = self.value_proj(key_value_input)  # (batch_size, seq_len, embedding_dim)
        value = value.view(batch_size, seq_len, self.num_heads, head_dim).transpose(
            1, 2
        )  # (batch_size, num_heads, seq_len, head_dim)

        # matrix multiply aka dot product produces scores between each position and all others showing affinity
        # this creates a square matrix where each position pair generates a score, the diagonal being with itself
        # large values in each signed direction mean position pairs are compatible or incompatible
        # NOTE: sqrt is used to keep values not too extreme for softmax, preventing saturation
        attn_scores = torch.matmul(query, key) / math.sqrt(head_dim)  # (batch_size, num_heads, seq_len, seq_len)

        # apply boolean mask to protect causality
        # aka each position not having scores associated with positions to the right of it
        causal_mask = mask[:seq_len, :seq_len]
        # negative infinity on the tensor entries marked as true means softmax later will turn these into virtually 0
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        # softmax converts raw attention scores to be a normalized probability distribution over all positions
        # it normalizes across the "key" dimension so each "query" position's attention weight sums to 1
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        # apply dropout to further regularize attention task
        attn_weights = self.attn_dropout(attn_weights)

        # Given the attention weights, does a weighted combination of each positions "value" representation
        attn_output = torch.matmul(attn_weights, value)  # (batch_size, num_heads, seq_len, head_dim)
        # Recombine heads by reshaping, making data layout contiguous first then condensing head_dim back to embedding_dim
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embedding_dim)
        )  # (batch_size, seq_len, embedding_dim)

        # Produce and return the projection of the heads through the linear layer
        # as a combined final output back to the original embedding_dim
        attn = self.out_proj(attn_output)  # (batch_size, seq_len, embedding_dim)
        return attn


class AttentionBlock(nn.Module):
    def __init__(
        self,
        dropout_rate: float,
        embedding_dim: int,
        num_heads: int = 1,
        ffn_dim: Optional[int] = None,
        ffn_activation: Literal["relu", "gelu"] = "relu",
        residual_from: Literal["input", "normed"] = "input",
        kv_from: Literal["normed", "input"] = "normed",
    ):

        super().__init__()

        # How the block is wired (CURRICULUM v4 and v5, issue #32). Each sublayer (attention, then the
        # feed-forward network) normalizes its input x, transforms it, and adds the result back:
        #     pre-LN, the defaults:   x + sublayer(LayerNorm(x))
        # That is the formula in the SASRec paper (Kang & McAuley 2018, Section III-C). The authors'
        # released code (kang205/SASRec) changes two wires, and each flag here flips one of them:
        #   residual_from="normed": add the output back onto LayerNorm(x) instead of x, for both
        #     sublayers. The residual stream is then re-normalized at every block.
        #   kv_from="input": attention's keys and values read x; only the queries read LayerNorm(x).
        # Both flipped is the authors' code. On ML-20M it collapses from 8 blocks, and so does
        # either flag on its own (CURRICULUM v5).
        self.is_residual_from_normed = residual_from == "normed"
        self.is_kv_from_input = kv_from == "input"

        # determine feed-forward network arguments
        ffn_hidden_dim = ffn_dim or embedding_dim
        ffn_activation_module = nn.GELU() if ffn_activation == "gelu" else nn.ReLU()

        # setup layers
        # attention layers
        self.norm_attn = nn.LayerNorm(normalized_shape=embedding_dim)
        self.attn = SelfAttention(embedding_dim=embedding_dim, dropout_rate=dropout_rate, num_heads=num_heads)
        self.dropout_attn = nn.Dropout(p=dropout_rate)
        # feed-forward network layers
        self.norm_ffn = nn.LayerNorm(normalized_shape=embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=ffn_hidden_dim, bias=True),
            ffn_activation_module,
            nn.Linear(in_features=ffn_hidden_dim, out_features=embedding_dim, bias=True),
        )
        self.dropout_ffn = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # forward attention: the queries always read LayerNorm(x)
        normed = self.norm_attn(x)
        y = self.attn(normed, mask=mask, key_value_input=x if self.is_kv_from_input else None)
        # apply attention dropout
        y = self.dropout_attn(y)
        # residual connection: add the input back so gradients can flow without going through attention.
        # pre-LN adds back x itself, so nothing inside the stack ever rescales the residual stream.
        x = (normed if self.is_residual_from_normed else x) + y
        # forward feed forward network with norm and dropout
        normed = self.norm_ffn(x)
        y = self.ffn(normed)
        y = self.dropout_ffn(y)
        # residual connection, wired the same way as the attention one
        return (normed if self.is_residual_from_normed else x) + y


class SASRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        max_seq_len: int,
        dropout_rate: float,
        embedding_dim: int,
        num_blocks: int = 2,
        num_heads: int = 1,
        ffn_dim: Optional[int] = None,
        ffn_activation: Literal["relu", "gelu"] = "relu",
        attention_block_residual_from: Literal["input", "normed"] = "input",
        attention_block_kv_from: Literal["normed", "input"] = "normed",
    ):

        super().__init__()

        # setup item and position embeddings tables to convert integers to dense vectors
        # +1 because item IDs are 1-indexed and index 0 is reserved for padding
        self.item_embedding_table = nn.Embedding(
            num_embeddings=num_items + 1, embedding_dim=embedding_dim, padding_idx=0
        )
        # max_seq_len - 1 because the shifted input/target split reduces sequence length by 1
        self.position_embedding_table = nn.Embedding(num_embeddings=max_seq_len - 1, embedding_dim=embedding_dim)
        self.dropout_embedding = nn.Dropout(p=dropout_rate)
        self.final_norm = nn.LayerNorm(embedding_dim)
        # Build attention blocks
        self.attn_blocks = nn.ModuleList(
            [
                AttentionBlock(
                    dropout_rate=dropout_rate,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    ffn_activation=ffn_activation,
                    residual_from=attention_block_residual_from,
                    kv_from=attention_block_kv_from,
                )
                for _ in range(num_blocks)
            ]
        )
        # Apply truncated normal weight initialization
        self.apply(lambda m: init_weights(m, embedding_dim))
        # create boolean mask to protect causality in forward pass
        # aka each position not having scores associated with positions to the right of it
        # NOTE: do not need to add other 2 dimensions to `ones`` creation due to broadcasting in masked_fill later
        mask = torch.triu(torch.ones((max_seq_len - 1, max_seq_len - 1)), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.size()
        # convert item IDs to dense vectors
        item_embeddings = self.item_embedding_table(x)  # (batch_size, seq_len, embedding_dim)
        # convert item positions to dense vectors
        # NOTE: Reversed positional indexing (Huang 2025, arxiv: 2504.09596):
        # With forward indexing e.g. a 5-item user predicts from position 4 while a
        # 200-item user predicts from position 199. the model must learn that
        # different position embeddings all mean "predict next item."
        # With reversed indexing, the prediction position is always embedding 0,
        # so the model consistently learns "how far back is this item."
        positions = torch.arange(seq_len - 1, -1, -1, device=x.device)
        position_embeddings = self.position_embedding_table(positions)  # (seq_len, embedding_dim)
        # add them together
        x = item_embeddings + position_embeddings  # automatic broadcast
        # apply dropout
        x = self.dropout_embedding(x)
        # process through entire module list of attention blocks
        for attn_block in self.attn_blocks:
            x = attn_block(x, mask=self.causal_mask)  # (batch_size, seq_len, embedding_dim)
        # final layer normalization since attention ends in residual add
        return self.final_norm(x)  # (batch_size, seq_len, embedding_dim)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


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
) -> Tuple[float, float]:
    """Full-catalog HR@K and NDCG@K on the validation or test split.

    mode="valid" ranks the second-to-last item given the train history.
    mode="test" ranks the last item given the train + valid history.

    The ground truth is ranked against every item in the catalog except the ones
    already in the user's input history. v1-v3 ranked it against 100 sampled
    negatives instead, which inflates metrics and can change which model looks
    better (Krichene & Rendle, KDD 2020).
    """
    hrs, ndcgs = [], []
    eval_max_len = max_len - 1  # required to match training

    # Pre-build all user sequences and ground truths
    all_seqs = []
    all_gts = []
    all_seens = []
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

    # Process in batches
    for start in range(0, num_users, eval_batch_size):
        end = min(start + eval_batch_size, num_users)
        batch_gts = all_gts[start:end]
        batch_seens = all_seens[start:end]

        # Batched forward pass, the last position is the prediction
        input_batch = torch.tensor(all_seqs[start:end], dtype=torch.long, device=device)
        pred_batch = model(input_batch)[:, -1, :]  # (batch_size, embedding_dim)

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


# ---------------------------------------------------------------------------
# Checkpoints: save and resume training runs
# ---------------------------------------------------------------------------
# Each run keeps its checkpoints in its own directory, named from the training arguments
# that change what the run computes. Re-running the same command finds that directory and
# continues from the latest checkpoint, after anything that kills the process: a crash,
# Ctrl-C, an out-of-memory kill, or a cloud preemption. A checkpoint holds the model, the
# optimizer, and the state of every random number generator training draws from, so a
# resumed run produces the same weights as one that was never interrupted.

# Arguments that don't change the weights a run produces, so they are left out of
# the run name. Leaving out --epochs means a finished run can be extended by
# re-running it with a larger --epochs (the learning rate is constant).
ARGS_EXCLUDED_FROM_RUN_HASH = (
    "data_dir",
    "device",
    "epochs",
    "eval_every",
    "topk",
    "checkpoint_dir",
    "checkpoint_every",
)


def checkpoint_run_dir(checkpoint_dir: str, script: str, args: Namespace) -> str:
    """Directory for this run's checkpoints, e.g. checkpoints/train_v4-ml-20m-seed42-3f9a1c2e."""
    config = {k: v for k, v in vars(args).items() if k not in ARGS_EXCLUDED_FROM_RUN_HASH}
    # Keep the first 8 hex characters (32 bits) of the hash: short enough to read in a path,
    # and long enough that two different configurations practically never share one.
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
    name = os.path.splitext(os.path.basename(script))[0]
    return os.path.join(checkpoint_dir, f"{name}-{args.dataset}-seed{args.seed}-{digest}")


def save_checkpoint(directory: str, epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer, args: Namespace):
    """Save the state after `epoch` completed epochs as epoch_XXXX.pt."""
    os.makedirs(directory, exist_ok=True)
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
        },
        "args": vars(args),  # for humans inspecting the file; not used on restore
    }
    path = os.path.join(directory, f"epoch_{epoch:04d}.pt")
    # Write to a temporary file, then rename. A crash or preemption mid-write leaves
    # only a .tmp file, which resume_checkpoint() ignores, never a truncated checkpoint.
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)


def resume_checkpoint(directory: str, model: nn.Module, optimizer: torch.optim.Optimizer) -> int:
    """Restore the latest checkpoint in `directory`, if any. Returns the number of completed epochs."""
    for path in sorted(glob.glob(os.path.join(directory, "epoch_*.pt")), reverse=True):
        try:
            # weights_only=False because the RNG states include NumPy arrays and tuples.
            # These are files this code wrote, not untrusted downloads.
            state = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:  # e.g. a file cut short by a crash outside the rename window
            print(f"WARNING: skipping unreadable checkpoint {path}: {e}")
            continue
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])  # moves optimizer state to the parameters' device
        rng = state["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        if rng["mps"] is not None and torch.backends.mps.is_available():
            torch.mps.set_rng_state(rng["mps"])
        print(f"Resumed from {path} ({state['epoch']} epochs completed)")
        return state["epoch"]
    print(f"No checkpoint in {directory}, starting from scratch")
    return 0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="nanoSASRec train v4")
    parser.add_argument("--dataset", type=str, default="ml-1m", choices=DATASETS, help="Dataset to use")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--batch-size", type=int, default=128, help="Number of examples per batch")
    parser.add_argument("--max-len", type=int, default=200, help="Max sequence length per user")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--dim", type=int, default=50, help="Embedding dimension size")
    parser.add_argument(
        "--ffn-dim", type=int, default=None, help="feed-forward network hidden dimension size (default: embedding dim)"
    )
    parser.add_argument(
        "--ffn-activation", type=str, default="relu", choices=["relu", "gelu"], help="feed-forward network activation"
    )
    parser.add_argument("--blocks", type=int, default=2, help="Number of attention blocks")
    parser.add_argument("--heads", type=int, default=1, help="Number of attention heads")
    # How each attention block is wired (CURRICULUM v4 and v5). The defaults are pre-LN,
    # x + sublayer(LayerNorm(x)). Flipping both flags gives the SASRec authors' released code.
    parser.add_argument(
        "--attention-block-residual-from",
        type=str,
        default="input",
        choices=["input", "normed"],
        help="What each sublayer's output is added back onto: its input x (pre-LN) or LayerNorm(x) "
        "(the SASRec authors' code)",
    )
    parser.add_argument(
        "--attention-block-kv-from",
        type=str,
        default="normed",
        choices=["normed", "input"],
        help="What attention's keys and values read: LayerNorm(x) like the queries (pre-LN) or the block input x "
        "(the SASRec authors' code)",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for optimizer during training")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs for training")
    parser.add_argument("--num-negatives", type=int, default=256, help="Number of negatives to sample for training")
    parser.add_argument("--topk", type=int, default=10, help="Cutoff for HR@K and NDCG@K")
    parser.add_argument("--eval-every", type=int, default=20, help="Evaluate every N epochs")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run computation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints", help="Root directory for resumable checkpoints"
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=5, help="Save a checkpoint every N epochs (0 to disable)"
    )
    args = parser.parse_args()
    print(f"\n{'=' * 60}")
    print(f"  {parser.description} (git: {get_git_sha()})")
    print(f"{'=' * 60}")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print(f"{'=' * 60}\n")

    # set seed across python, numpy, and torch
    set_seed(args.seed)

    # Determine device
    device = torch.device(args.device)
    if torch.cuda.is_available() and args.device == "cpu":
        print("WARNING: CUDA is available but running on CPU. Use --device cuda for faster training.")

    # Load pre-processed dataset with train, validation, test splits
    user_train, user_valid, user_test, num_users, num_items, item_id_map = load_data(args.data_dir, args.dataset)
    print("Dataset Lengths:", len(user_train), len(user_valid), len(user_test))
    # load titles for future demos
    titles = load_item_titles(args.data_dir, args.dataset, item_id_map)

    # Init SASRec model with hyperparams
    model = SASRec(
        num_items=num_items,
        max_seq_len=args.max_len,
        dropout_rate=args.dropout,
        embedding_dim=args.dim,
        num_blocks=args.blocks,
        num_heads=args.heads,
        ffn_dim=args.ffn_dim,
        ffn_activation=args.ffn_activation,
        attention_block_residual_from=args.attention_block_residual_from,
        attention_block_kv_from=args.attention_block_kv_from,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    model = model.to(device)

    # create and setup Adam optimizer
    # betas=(0.9, 0.98) from original Transformer paper, PyTorch default is (0.9, 0.999)
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    batches_per_epoch = num_users // args.batch_size
    print("Estimate batches per epoch:", batches_per_epoch)

    # Resume from this run's latest checkpoint, if there is one (e.g. after Modal preempts the container).
    # Same arguments -> same run directory, so re-running the same command continues where it left off.
    run_checkpoint_dir = checkpoint_run_dir(args.checkpoint_dir, parser.prog, args)
    start_epoch = resume_checkpoint(run_checkpoint_dir, model, optimizer)

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # enter train mode for model to update model params
        model.train()
        # initialize epoch loss to log every iteration
        epoch_loss = 0.0
        # iterate over batches per epoch and update model parameters given loss
        for batch_idx in range(batches_per_epoch):
            # Zero out last loops gradient accumulations
            optimizer.zero_grad()

            # Sampled input user sequences, causal preserving target positives, and user-level negatives for loss calculation
            # input and positive sequence arrays are max_len - 1 length due to autoregressive learning task setup
            # each row in batch_size is a sampled user
            input_seq, pos_seq, neg_seq = sample_batch(
                user_train,
                num_users=num_users,
                num_items=num_items,
                max_len=args.max_len,
                batch_size=args.batch_size,
                num_negs=args.num_negatives,
            )

            # Convert numpy int64 arrays to torch LongTensors
            input_seq = torch.as_tensor(input_seq).to(device)  # (batch_size, seq_len)
            pos_seq = torch.as_tensor(pos_seq).to(device)  # (batch_size, seq_len)
            neg_seq = torch.as_tensor(neg_seq).to(device)  # (batch_size, num_negs)

            # Determine embeddings for input, positive and negative sequences
            output_embedding_seq = model.forward(input_seq)  # (batch_size, seq_len, embedding_dim)
            pos_embedding_seq = model.item_embedding_table(pos_seq)  # (batch_size, seq_len, embedding_dim)
            neg_embedding_seq = model.item_embedding_table(neg_seq)  # (batch_size, num_negs, embedding_dim)

            # Calculate the scores aka logits with the processed input sequence for the positive and negative examples each
            pos_logits_seq = torch.multiply(output_embedding_seq, pos_embedding_seq).sum(
                dim=-1
            )  # (batch_size, seq_len)
            neg_logits_seq = torch.matmul(
                output_embedding_seq, neg_embedding_seq.transpose(1, 2)
            )  # size(batch_size, seq_len, num_negs)

            # Calculate Sampled Softmax Loss (SSML)
            # Loss: -log( exp(score_pos) / (exp(score_pos) + sum(exp(score_neg))) )
            # NOTE: the version used below is numerically unstable solution
            # if any negative scores are large (say > 80), exp(80) overflows float32.
            # This won't happen in early training but could in later epochs
            # as the model becomes more confident
            neg_seq_logits_exp_sum = neg_logits_seq.exp().sum(dim=-1)  # (batch_size, seq_len)
            pos_seq_logits_exp = pos_logits_seq.exp()  # (batch_size, seq_len)
            loss = -torch.log(
                pos_seq_logits_exp / (pos_seq_logits_exp + neg_seq_logits_exp_sum)
            )  # size(batch_size, seq_len)
            # Below would be the more stable version than above but not as easy to read
            # -----
            # stack positive and negatives together: (batch_size, seq_len, 1 + num_negs)
            # logits_seq = torch.cat([pos_logits_seq.unsqueeze(dim=-1), neg_logits_seq], dim=-1)
            # logsumexp across candidates, then subtract the positive score
            # loss = torch.logsumexp(logits_seq, dim=-1) - pos_logits_seq
            # -----

            # Apply mask to remove sequences that have padding values as positive targets
            # which are invalid targets and add noise to the loss signal
            # Example:
            # original (padded):  [0, 0, 0, A, B, C]
            #             input:  [0, 0, 0, A, B]     (padded[:-1])
            #            target:  [0, 0, A, B, C]     (padded[1:])
            mask = pos_seq != 0  # boolean True where pad value `0` is not present
            loss = loss * mask
            # preserves averaging with variable padded sequences over only valid positions
            ssml = loss.sum() / mask.sum()
            # Conduct backwards propagation and update parameters
            ssml.backward()
            optimizer.step()
            # log current batch iteration loss value
            epoch_loss += ssml.item()

        # log loss and periodically do evaluation
        if (epoch + 1) % args.eval_every == 0 or (epoch + 1) == args.epochs:
            # perform evaluation for HR@K and NDCG@K
            model.eval()
            with torch.no_grad():
                hr, ndcg = evaluate(
                    model=model,
                    user_train=user_train,
                    user_valid=user_valid,
                    user_test=user_test,
                    num_users=num_users,
                    num_items=num_items,
                    max_len=args.max_len,
                    device=device,
                    mode="valid",
                    topk=args.topk,
                )
            # log metrics and loss so far
            avg_loss = epoch_loss / batches_per_epoch
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
            print(
                f"[{ts}] [Epoch {epoch + 1}/{args.epochs}] "
                f"Avg loss: {avg_loss:.4f} (~{math.exp(-avg_loss):.1%} correct) | "
                f"HR@{args.topk}: {hr:.4f} NDCG@{args.topk}: {ndcg:.4f}"
            )
        else:
            # log loss so far
            avg_loss = epoch_loss / batches_per_epoch
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
            print(
                f"[{ts}] [Epoch {epoch + 1}/{args.epochs}] "
                f"Avg loss: {avg_loss:.4f} (~{math.exp(-avg_loss):.1%} correct)"
            )

        # save everything needed to resume from here, including the RNG states
        if args.checkpoint_every and ((epoch + 1) % args.checkpoint_every == 0 or (epoch + 1) == args.epochs):
            save_checkpoint(run_checkpoint_dir, epoch + 1, model, optimizer, args)

    # test set evaluation
    model.eval()
    with torch.no_grad():
        test_hr, test_ndcg = evaluate(
            model,
            user_train,
            user_valid,
            user_test,
            num_users,
            num_items,
            args.max_len,
            device,
            mode="test",
            topk=args.topk,
        )
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(
        f"[{ts}] [Epochs {args.epochs}] Test metrics: HR@{args.topk}: {test_hr:.4f} NDCG@{args.topk}: {test_ndcg:.4f}"
    )

    # Run playful demo to showcase predictions
    if titles:
        # find a user with decent history
        eligible_users = [u for u in range(1, num_users + 1) if len(user_train[u]) >= 20]
        demo_user = random.choice(eligible_users) if eligible_users else None
        # run demo
        if demo_user:
            demo(demo_user, model, user_train, user_valid, user_test, num_items, args.max_len, device, titles)


if __name__ == "__main__":
    main()
