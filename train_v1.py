"""
train_v1.py
"""
import argparse
import json
from typing import Dict, Tuple, List
import random
import math

import numpy as np
import torch
import torch.nn as nn

DATASETS = ("ml-1m", "ml-20m", "beauty")

def load_data(data_dir: str, dataset: str) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, List[int]], int, int, Dict[int, str]]:
    """Load preprocessed dataset split from disk.

    Returns: user_train, user_valid, user_test, num_users, num_items, item_id_map
    """
    dataset_split_filepath = f"{data_dir}/{dataset}/split.json"
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

def sample_batch(user_train: Dict[int, List[int]], num_users: int, num_items: int, maxlen: int, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Select random users of size batch_size
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
            # TODO: given np array is init with 0s this is redudant, could insert at correct index
            num_pads = maxlen - seqlen
            pad_seq = [0] * num_pads
            user_seq = pad_seq + user_seq
        # create input and target sequences
        # create additional sub-sequences from a single sequence
        # ideally done by shifting the input sequence left and lining up
        input_seq, target_seq = user_seq[:-1], user_seq[1:]  # each seq is maxlen - 1
        # select random negatives from unseen items throuhgout entire sequence history
        # TODO: can be optimized to be O(1) using `random.randint(1, num_items)` and checking if seen
        item_neg_pool = set(range(1, num_items + 1)) - set(user_seq)
        random_neg_seq = random.sample(list(item_neg_pool), k=len(target_seq))
        # move list of ints into numpy arrays
        input_seq_batch[i, :] = input_seq
        target_pos_seq_batch[i, :] = target_seq
        target_neg_seq_batch[i, :] = random_neg_seq
    return input_seq_batch, target_pos_seq_batch, target_neg_seq_batch


class SASRec(nn.Module):

    def __init__(self, num_items:int, max_seq_len:int, dropout_rate:float, embedding_dim:int, num_blocks: int = 2, num_heads: int = 1):

        super().__init__()

        # args
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        
        # Setup item and position embeddings tables to convert integers to dense vectors
        self.item_embedding_table = nn.Embedding(num_embeddings=num_items + 1, embedding_dim=embedding_dim, padding_idx=0)
        self.pos_embeding_table = nn.Embedding(num_embeddings=max_seq_len - 1, embedding_dim=embedding_dim)

        self.norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.query_proj = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False)
        self.key_proj = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False)
        self.value_proj = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False)
        self.out_proj = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True)
        
        # blocks = []
        # for i in range(num_blocks):
        #     blocks.append(nn.LayerNorm(normalized_shape=(seq_len, embedding_dim)))
        #     blocks.append(nn.Dropout(p=dropout_rate))
        # self.attn_blocks = nn.ModuleList(blocks)
    
    def self_attention(self, x):
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
        # this creates a square matrix where each position pair generates a score, the diagnol being with itself
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

def main():
    parser = argparse.ArgumentParser(description="nanoSASRec train v1")
    parser.add_argument("--dataset", type=str, default="ml-1m",
                        choices=DATASETS,
                        help="Dataset to use")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Root data directory")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Number of examples per batch")
    parser.add_argument("--maxlen", type=int, default=50,
                        help="Max sequence length per user")
    args = parser.parse_args()

    # Load pre-processed dataset with train, validation, test splits
    user_train, user_valid, user_test, num_users, num_items, item_id_map = load_data(args.data_dir, args.dataset)
    input_seq, pos_seq, neg_seq = sample_batch(user_train, num_users=num_users, num_items=num_items, maxlen=args.maxlen, batch_size=args.batch_size)
    print(input_seq.shape, pos_seq.shape, neg_seq.shape)

if __name__ == "__main__":
    main()