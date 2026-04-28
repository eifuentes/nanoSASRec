"""
train_v1.py — vanilla SASRec + BCE

Single-file implementation of Self-Attentive Sequential Recommendation (SASRec)
with binary cross-entropy loss and negative sampling.

Paper: Kang & McAuley, 2018 (arxiv: 1808.09781)
"""
import argparse
import json
import math
import os
import random
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

DATASETS = ("ml-1m", "ml-20m", "beauty")

def load_data(data_dir: str, dataset: str) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, List[int]], int, int, Dict[int, str]]:
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

def sample_batch(user_train: Dict[int, List[int]], num_users: int, num_items: int, maxlen: int, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Select random users without replacement of size batch_size
    # NOTE: in the original SASRec they sample users with replacement
    batch_users = random.sample(range(1, num_users + 1), k=batch_size)
    # allocate input, target, and negative sequence batch matrices
    input_seq_batch = np.zeros((batch_size, maxlen - 1), int)
    target_pos_seq_batch = np.zeros((batch_size, maxlen - 1), int)
    target_neg_seq_batch = np.zeros((batch_size, maxlen - 1), int)
    # extract user sequence and truncate or pad sequence to maxlen
    for i, user in enumerate(batch_users):
        user_seq = user_train[user]
        seqlen = len(user_seq)
        if seqlen >= maxlen:
            # User sequences are temporally ordered from oldest to newest
            # For these models its best practice to keep the most recent interactions
            start_idx = seqlen - maxlen
            user_seq = user_seq[start_idx:]
        else:
            # 0 is reserved value for padding values
            # TODO: given np array is init with 0s this is redundant, could insert at correct index
            num_pads = maxlen - seqlen
            pad_seq = [0] * num_pads
            user_seq = pad_seq + user_seq
        # create input and target sequences
        # create additional sub-sequences from a single sequence
        # ideally done by shifting the input sequence left and lining up
        input_seq, target_seq = user_seq[:-1], user_seq[1:]  # each seq is maxlen - 1
        # select random negatives from unseen items throughout entire sequence history
        # TODO: can be optimized to be O(1) using `random.randint(1, num_items)` and checking if seen
        item_neg_pool = set(range(1, num_items + 1)) - set(user_seq)
        random_neg_seq = random.sample(list(item_neg_pool), k=len(target_seq))
        # move list of ints into numpy arrays
        input_seq_batch[i, :] = input_seq
        target_pos_seq_batch[i, :] = target_seq
        target_neg_seq_batch[i, :] = random_neg_seq
    return input_seq_batch, target_pos_seq_batch, target_neg_seq_batch


class SelfAttention(nn.Module):

    def __init__(self, embedding_dim:int, num_heads: int = 1):
        
        super().__init__()

        # args
        self.num_heads = num_heads

        # setup the attention projections
        self.query_proj = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False)
        self.key_proj = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False)
        self.value_proj = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False)
        self.out_proj = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scaled Dot-Product Self-Attention.

        Args:
            x (torch.Tensor): input tensor of size(batch_size, seq_len, embedding_dim)

        Returns:
            torch.Tensor: attended input tensor of size(batch_size, seq_len, embedding_dim)
        """
        # input tensor x is of shape (batch_size, seq_len, embedding_dim)
        batch_size, seq_len, embedding_dim = x.size()
        
        # infer attention head dimensions from argument num_heads, should be cleanly divisible
        head_dim = embedding_dim // self.num_heads
        
        # project query, key, value and calculate attention scores
        # each linear layer contains all weights for all heads for efficency and later breaks them out via reshaping

        # query linear layer transforms each position into "this is what I am looking for?"
        query = self.query_proj(x)  # (batch_size, seq_len, embedding_dim)
        query = query.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        
        # key linear layer transforms each position into "this is what I advertise about myself so queries can find me?"
        key = self.key_proj(x)  # (batch_size, seq_len, embedding_dim)
        key = key.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        # transpose key for future matrix multiply so dimensions line up with query tensor
        key = key.transpose(-2, -1)  # (batch_size, num_heads, head_dim, seq_len)
        
        # value linear layer transforms each position into "this is the information I pass along when attended to?"
        value = self.value_proj(x)  # (batch_size, seq_len, embedding_dim)
        value = value.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
       
        # matrix multiply aka dot product produces scores between each position and all others showing affinity
        # this creates a square matrix where each position pair generates a score, the diagonal being with itself
        # large values in each signed direction mean position pairs are compatible or incompatible
        # NOTE: sqrt is used to keep values not too extreme for softmax, preventing saturation
        attn_scores = torch.matmul(query, key) / math.sqrt(head_dim)  # (batch_size, num_heads, seq_len, seq_len)

        # create and apply boolean mask to protect causality
        # aka each position not having scores associated with positions to the right of it
        # NOTE: do not need to add other 2 dimensions to ones creation due to broadcasting in masked_fill
        # TODO: Creating causal mask on every forward pass, move to registered buffer in __init__ instead
        mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device), diagonal=1).bool()
        # negative infinity on the tensor entries marked as true means softmax later will turn these into virtually 0
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        # softmax converts raw attention scores to be a normalized probability distribution over all positions
        # it normalizes across the "key" dimension so each "query" position's attention weight sums to 1
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)

        # Given the attention weights, does a weighted combination of each positions "value" representation
        attn_output = torch.matmul(attn_weights, value)  # (batch_size, num_heads, seq_len, head_dim)
        # Recombine heads by reshaping, making data layout contiguous first then condensing head_dim back to embedding_dim
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embedding_dim)  # (batch_size, seq_len, embedding_dim)

        # Produce and return the projection of the heads through the linear layer as a combined final output back to the original embedding_dim
        attn = self.out_proj(attn_output)  # (batch_size, seq_len, embedding_dim)
        return attn

class AttentionBlock(nn.Module):

    def __init__(self, dropout_rate:float, embedding_dim:int, num_heads: int = 1):

        super().__init__()

        # setup layers
        # attention layers
        self.norm_attn = nn.LayerNorm(normalized_shape=embedding_dim)
        self.attn = SelfAttention(embedding_dim=embedding_dim, num_heads=num_heads)
        self.dropout_attn = nn.Dropout(p=dropout_rate)
        # feed-forward network layers
        self.norm_ffn = nn.LayerNorm(normalized_shape=embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True),
        )
        self.dropout_ffn = nn.Dropout(p=dropout_rate)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # forward attention
        y = self.norm_attn(x)
        y = self.attn(y)
        y = self.dropout_attn(y)
        x = x + y  # residual connection: add original input back so gradients can flow through unchanged
        # forward feed forward network
        y = self.norm_ffn(x)
        y = self.ffn(y)
        y = self.dropout_ffn(y)
        return x + y  # residual connections


class SASRec(nn.Module):

    def __init__(self, num_items:int, max_seq_len:int, dropout_rate:float, embedding_dim:int, num_blocks: int = 2, num_heads: int = 1):

        super().__init__()
        
        # setup item and position embeddings tables to convert integers to dense vectors
        # +1 because item IDs are 1-indexed and index 0 is reserved for padding
        self.item_embedding_table = nn.Embedding(num_embeddings=num_items + 1, embedding_dim=embedding_dim, padding_idx=0)
        # max_seq_len - 1 because the shifted input/target split reduces sequence length by 1
        self.position_embedding_table = nn.Embedding(num_embeddings=max_seq_len - 1, embedding_dim=embedding_dim)
        self.dropout_embedding = nn.Dropout(p=dropout_rate)
        self.final_norm = nn.LayerNorm(embedding_dim)
        # Build attention blocks
        self.attn_blocks = nn.ModuleList([
            AttentionBlock(dropout_rate=dropout_rate, embedding_dim=embedding_dim, num_heads=num_heads)
            for _ in range(num_blocks)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.size()
        # convert item IDs to dense vectors
        item_embeddings = self.item_embedding_table(x)  # (batch_size, seq_len, embedding_dim)
        # convert item positions to dense vectors
        positions = torch.arange(seq_len, device=x.device)
        position_embeddings = self.position_embedding_table(positions)  # (seq_len, embedding_dim)
        # add them together
        embeddings = item_embeddings + position_embeddings  # automatic broadcast
        # apply dropout
        embeddings = self.dropout_embedding(embeddings)
        # process through entire module list of attention blocks
        x = embeddings
        for attn_block in self.attn_blocks:
            x = attn_block(x)  # (batch_size, seq_len, embedding_dim)
        # final layer normalization since attention ends in residual add 
        return self.final_norm(x)  # (batch_size, seq_len, embedding_dim)
    
    def predict(self, input_seq: torch.Tensor, candidate_seqs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        # input and candidate sequence size(batch_size, seq_len)
        output_seq = self.forward(input_seq)  # (batch_size, seq_len, embedding_dim)
        candidate_scores = []
        for candidate_seq in candidate_seqs:
            candidate_embeddings = self.item_embedding_table(candidate_seq)  # (batch_size, seq_len, embedding_dim)
            # score each position's model output against its candidate item embedding
            # element-wise multiply (not matmul) because we want one score per position,
            # not all-pairs: e.g. position 3's output scored only against position 3's candidate
            scores = torch.multiply(output_seq, candidate_embeddings).sum(dim=-1)  # (batch_size, seq_len)
            candidate_scores.append(scores)
        return tuple(candidate_scores) # (batch_size, seq_len)

def evaluate(model: SASRec, user_train: Dict[int, List[int]], user_valid: Dict[int, List[int]], user_test: Dict[int, List[int]], num_users: int, num_items: int, maxlen: int, device: torch.device, mode: str = "valid", topk:int = 10) -> Tuple[float, float]:
    hrs, ndcgs = [], []
    for user in range(1, num_users + 1):
        # generate user sequence based on mode
        user_seq = list(user_train[user])  # copy
        if mode == "test":
            # for test mode, add validation sequence to train
            user_seq = user_seq + user_valid[user]

        # pad to max sequence length
        seqlen = len(user_seq)
        max_len = maxlen - 1  # required to match training
        if seqlen >= max_len:
            # User sequences are temporally ordered from oldest to newest
            # For these models its best practice to keep the most recent interactions
            start_idx = seqlen - max_len
            user_seq = user_seq[start_idx:]
        else:
            # 0 is reserved value for padding values
            num_pads = max_len - seqlen
            pad_seq = [0] * num_pads
            user_seq = pad_seq + user_seq

        # convert to tensors and do model forward pass extracting last position as prediction
        input_seq = torch.tensor(user_seq, dtype=torch.long).expand(1, -1).to(device)
        output_seq = model.forward(input_seq)
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
        # this is addressed in train_v2.py+ with sampled softmax and full-catalog evaluation
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

def demo(user, model, user_train, user_valid, user_test, num_items, maxlen, device, titles):
    def name(item_id):
        return titles.get(item_id, f"item_{item_id}")

    # build input sequence (train + valid, same as test eval)
    user_seq = list(user_train[user]) + user_valid[user]
    max_len = maxlen - 1
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
        scores[item - 1] = float('-inf')  # -1 because items are 1-indexed, scores are 0-indexed

    # top 10
    top10 = (scores.topk(10).indices + 1).cpu().tolist()  # back to 1-indexed

    print(f"\n  Prediction demo (user {user}):")
    print(f"    Recent history:  {', '.join(name(i) for i in user_train[user][-5:])}")
    print(f"    Ground truth:    {name(ground_truth)}")
    print(f"    Model top-10:")
    for rank, item_id in enumerate(top10, 1):
        marker = " <- hit" if item_id == ground_truth else ""
        print(f"      {rank:2d}. {name(item_id)}{marker}")
    print()


def main():
    parser = argparse.ArgumentParser(description="nanoSASRec train v1")
    parser.add_argument("--dataset", type=str, default="ml-1m",
                        choices=DATASETS,
                        help="Dataset to use")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Root data directory")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Number of examples per batch")
    parser.add_argument("--maxlen", type=int, default=200,
                        help="Max sequence length per user")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate")
    parser.add_argument("--dim", type=int, default=50,
                        help="Embedding dimension size")
    parser.add_argument("--blocks", type=int, default=2,
                        help="Number of attention blocks")
    parser.add_argument("--heads", type=int, default=1,
                        help="Number of attention heads")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for optimizer during training")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of epochs for training")
    parser.add_argument("--topk", type=int, default=10,
                        help="Cutoff for HR@K and NDCG@K")
    parser.add_argument("--eval-every", type=int, default=20,
                        help="Evaluate every N epochs")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run computation")
    args = parser.parse_args()

    # Determine device
    device = torch.device(args.device)
    if torch.cuda.is_available() and args.device == "cpu":
        print("WARNING: CUDA is available but running on CPU. Use --device cuda for faster training.")

    # numerical stability for sigmoid in loss function
    eps = 1e-7
    # Load pre-processed dataset with train, validation, test splits
    user_train, user_valid, user_test, num_users, num_items, item_id_map = load_data(args.data_dir, args.dataset)
    print("Dataset Lengths:", len(user_train), len(user_valid), len(user_test))
    # load titles for future demos
    titles = load_item_titles(args.data_dir, args.dataset, item_id_map)

    # Init SASRec model with hyperparams
    model = SASRec(num_items=num_items, max_seq_len=args.maxlen, dropout_rate=args.dropout, embedding_dim=args.dim, num_blocks=args.blocks, num_heads=args.heads)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    model = model.to(device)
    # create and setup Adam optimizer
    # betas=(0.9, 0.98) from original Transformer paper, PyTorch default is (0.9, 0.999)
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    batches_per_epoch = num_users // args.batch_size
    print("Est. batches per epoch:", batches_per_epoch)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in range(batches_per_epoch):
            # Zero out last loops gradient accumulations
            optimizer.zero_grad()
            
            # Sample input, target positives, and negatives for BCE loss sequences
            input_seq, pos_seq, neg_seq = sample_batch(user_train, num_users=num_users, num_items=num_items, maxlen=args.maxlen, batch_size=args.batch_size)
            # print("Input Shapes:", input_seq.shape, pos_seq.shape, neg_seq.shape)

            # Compute positive and negative sequence probabilities
            input_seq, pos_seq, neg_seq = torch.as_tensor(input_seq).to(device), torch.as_tensor(pos_seq).to(device), torch.as_tensor(neg_seq).to(device)
            pos_seq_scores, neg_seq_scores = model.predict(input_seq, candidate_seqs=(pos_seq, neg_seq))
            # numerical stability: sigmoid can output values at exactly 0.0 or 1.0
            # due to floating point, and log(0) = -inf which poisons the loss.
            # clamp keeps probabilities in (eps, 1-eps) so log is always finite.
            # in practice, use nn.BCEWithLogitsLoss which avoids this entirely
            # by never computing log(sigmoid(x)) directly.
            pos_seq_probas, neg_seq_probas = pos_seq_scores.sigmoid().clamp(eps, 1 - eps), neg_seq_scores.sigmoid().clamp(eps, 1 - eps)
            # print("Proba shapes:", pos_seq_probas.size(), neg_seq_probas.size())
            
            # Calculate binary cross entropy (BCE) loss
            # BCE: -log(P) for positives (push toward 1), -log(1-P) for negatives (push toward 0)
            loss = pos_seq_probas.log().neg() + (1 - neg_seq_probas).log().neg()
            # print("Loss:", loss.size())
            # Apply mask to remove sequences that have padding values as positive targets
            # which are invalid targets and add noise to the loss signal
            # Example:
            # original (padded):  [0, 0, 0, A, B, C]
            #             input:  [0, 0, 0, A, B]     (padded[:-1])
            #            target:  [0, 0, A, B, C]     (padded[1:])
            mask = (pos_seq != 0)  # boolean True where pad value `0` is not present
            loss = loss * mask
            # preserves averaging with variable padded sequences over only valid positions
            bce = loss.sum() / mask.sum()
            # Conduct backwards propagation and update parameters
            bce.backward()
            optimizer.step()
            # log current batch iteration loss value
            epoch_loss += bce.item()

        # log loss and periodically do evaluation
        if (epoch + 1) % args.eval_every == 0 or (epoch + 1) == args.epochs:
            # perform evaluation for HR@K and NDCG@K
            model.eval()
            with torch.no_grad():
                hr, ndcg = evaluate(model=model, user_train=user_train, user_valid=user_valid, user_test=user_test, num_users=num_users, num_items=num_items, maxlen=args.maxlen, device=device, mode="valid", topk=args.topk)
            # log metrics and loss so far
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
            print(f"[{ts}] [Epoch {epoch + 1}/{args.epochs}] Avg BCE loss: {epoch_loss / batches_per_epoch:.4f} | HR@{args.topk}: {hr:.4f} NDCG@{args.topk}: {ndcg:.4f}")
        else:
            # log loss so far
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
            print(f"[{ts}] [Epoch {epoch + 1}/{args.epochs}] Avg BCE loss: {epoch_loss / batches_per_epoch:.4f}")

    # test set evaluation
    model.eval()
    with torch.no_grad():
        test_hr, test_ndcg = evaluate(model, user_train, user_valid, user_test, num_users, num_items, args.maxlen, device, mode="test", topk=args.topk)
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] [Epochs {args.epochs}] Test metrics: HR@{args.topk}: {test_hr:.4f} NDCG@{args.topk}: {test_ndcg:.4f}")

    # Run playful demo to showcase predictions
    if titles:
        # find a user with decent history
        demo_user = None
        for u in range(1, num_users + 1):
            if len(user_train[u]) >= 20:
                demo_user = u
                break
        # run demo
        if demo_user:
            demo(demo_user, model, user_train, user_valid, user_test, num_items, args.maxlen, device, titles)

if __name__ == "__main__":
    main()