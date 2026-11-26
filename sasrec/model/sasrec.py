"""
sasrec.py - SASRec model architecture for nanoSASRec v4+

Shared model components used by train_v4.py and later versions.
Architecture is identical to v1-v3 with these refinements:
    - Weight init: truncated normal with std=1/sqrt(d) (Zhai et al. (2024))
    - Reversed positional embeddings (Huang 2025, arxiv: 2504.09596)
    - Causal mask as registered buffer (created once, not per forward pass)
    - Configurable feed-forward network hidden dim and activation function
    - Attention weight dropout (standard in Transformer implementations, e.g. tensor2tensor)
    - Residual scaling via ReZero (Bachlechner et al. 2020) for stable depth scaling
    - Relative position bias (Raffel et al. 2020, Zhai et al. 2024) with log-bucketed distances
"""

import math
from typing import Literal, Optional

import torch
import torch.nn as nn


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


def bucket_relative_positions(relative_positions: torch.Tensor, num_buckets: int = 32) -> torch.Tensor:
    """Convert relative position distances to bucket indices.

    First half of buckets are linear (one per distance for fine-grained recency).
    Second half are logarithmically spaced (coarse for distant items).
    """
    # with num_buckets=32, this is 16.
    # Distances 0-15 each get their own dedicated bucket.
    # Distance 0 → bucket 0, distance 1 → bucket 1, ..., distance 15 → bucket 15
    max_exact = num_buckets // 2
    # determine max_distance from the tensor of relative distances itself
    # guard for sequences where max_distance is less than max_exact
    max_distance = max(relative_positions.max().item(), max_exact + 1)
    # only care about non-negative distances (causal: i attends to j where j <= i)
    # negative distances mean "future positions" which causal masking blocks anyway.
    # clamp them to 0 so we don't get negative indices or log-of-negative errors.
    # example: the matrix [[0,-1,-2],[1,0,-1],[2,1,0]] becomes [[0,0,0],[1,0,0],[2,1,0]]
    relative_positions = relative_positions.clamp(min=0)
    # small distances: linear bucket (one per distance)
    # create boolean mask.
    # True wherever the distance is small enough for a linear bucket.
    is_small = relative_positions < max_exact
    # avoid log(0) in the computation even though torch.where won't select it
    safe_positions = relative_positions.clamp(min=1)
    # large distances: logarithmic bucket
    log_buckets = (
        max_exact
        + (  # offset past the linear buckets
            # normalize distances relative to where the log region starts
            # then applies a log transform
            torch.log(safe_positions.float() / max_exact)
            # normalize so that max_distance maps to 1.0
            / math.log(max_distance / max_exact)
            # scale to fill the remaining N buckets
            * (num_buckets - max_exact)
        ).long()
    )
    # caps at num_buckets - 1 (the last valid bucket index)
    log_buckets = log_buckets.clamp(max=num_buckets - 1)
    # selects linear where small, log where large
    buckets = torch.where(is_small, relative_positions, log_buckets)
    return buckets


def bucket_temporal_distances(relative_timestamps: torch.Tensor, num_buckets: int = 32) -> torch.Tensor:
    """Convert relative timestamp timedeltas to bucket indices.

    First half of buckets are linear (one per timedelta for fine-grained recency).
    Second half are logarithmically spaced (coarse for larger timedeltas).
    """
    # with num_buckets=32, this is 16.
    # Distances 0-15 each get their own dedicated bucket.
    # Distance 0 → bucket 0, timedelta 1 → bucket 1, ..., timedelta 15 → bucket 15
    max_exact = num_buckets // 2
    # determine max_timedelta as a lookback window in seconds
    # picked 30 days which covers most user behavior windows
    max_timedelta = 86400 * 30  # 30 days in seconds
    # only care about non-negative timedeltas (causal: i attends to j where j <= i)
    # negative timedeltas mean "future timestamps" which causal masking blocks anyway.
    # clamp them to 0 so we don't get negative indices or log-of-negative errors.
    # example: the matrix [[0,-1,-2],[1,0,-1],[2,1,0]] becomes [[0,0,0],[1,0,0],[2,1,0]]
    relative_timedelta = relative_timestamps.clamp(min=0)
    # small timedeltas: linear bucket (one per timedelta)
    # create boolean mask.
    # True wherever the timedelta is small enough for a linear bucket.
    is_small = relative_timedelta < max_exact
    # avoid log(0) in the computation even though torch.where won't select it
    safe_positions = relative_timedelta.clamp(min=1)
    # large timedeltas: logarithmic bucket
    log_buckets = (
        max_exact
        + (  # offset past the linear buckets
            # normalize timedeltas relative to where the log region starts
            # then applies a log transform
            torch.log(safe_positions.float() / max_exact)
            # normalize so that max_timedelta maps to 1.0
            / math.log(max_timedelta / max_exact)
            # scale to fill the remaining N buckets
            * (num_buckets - max_exact)
        ).long()
    )
    # caps at num_buckets - 1 (the last valid bucket index)
    log_buckets = log_buckets.clamp(max=num_buckets - 1)
    # selects linear where small, log where large
    buckets = torch.where(is_small, relative_timedelta, log_buckets)
    return buckets


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

    def forward(self, x: torch.Tensor, mask: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Scaled Dot-Product Self-Attention.

        Args:
            x (torch.Tensor): input tensor of size(batch_size, seq_len, embedding_dim)
            mask (torch.Tensor): mask to preserve causal nature of attention scores
            bias (torch.Tensor, optional): relative bias of size(1, num_heads, seq_len, seq_len)

        Returns:
            torch.Tensor: attended input tensor of size(batch_size, seq_len, embedding_dim)
        """
        # input tensor x is of shape (batch_size, seq_len, embedding_dim)
        batch_size, seq_len, embedding_dim = x.size()

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
        key = self.key_proj(x)  # (batch_size, seq_len, embedding_dim)
        key = key.view(batch_size, seq_len, self.num_heads, head_dim).transpose(
            1, 2
        )  # (batch_size, num_heads, seq_len, head_dim)
        # transpose key for future matrix multiply so dimensions line up with query tensor
        key = key.transpose(-2, -1)  # (batch_size, num_heads, head_dim, seq_len)

        # value linear layer transforms each position into "this is the information I pass along when attended to?"
        value = self.value_proj(x)  # (batch_size, seq_len, embedding_dim)
        value = value.view(batch_size, seq_len, self.num_heads, head_dim).transpose(
            1, 2
        )  # (batch_size, num_heads, seq_len, head_dim)

        # matrix multiply aka dot product produces scores between each position and all others showing affinity
        # this creates a square matrix where each position pair generates a score, the diagonal being with itself
        # large values in each signed direction mean position pairs are compatible or incompatible
        # NOTE: sqrt is used to keep values not too extreme for softmax, preventing saturation
        attn_scores = torch.matmul(query, key) / math.sqrt(head_dim)  # (batch_size, num_heads, seq_len, seq_len)

        # optionally add relative position and/or temporal bias to attention scores.
        # unlike absolute position embeddings (added to item embeddings before attention),
        # this injects position information directly into the attention score matrix.
        # this keeps content signal (Q·K similarity) separate from position signal,
        # letting the model independently learn "how similar are these items" and
        # "how far apart are they in the sequence."
        # (Raffel et al. 2020 - T5, Zhai et al. 2024 - HSTU)
        if bias is not None:
            attn_scores = attn_scores + bias

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
        scale_residual: bool = False,
    ):

        super().__init__()

        # determine feed-forward network arguments
        ffn_hidden_dim = ffn_dim or embedding_dim
        ffn_activation_module = nn.GELU() if ffn_activation == "gelu" else nn.ReLU()

        # setup learnable parameters
        self.scale_residual = scale_residual
        # See for detail - ReZero (Bachlechner et al. 2020)
        if self.scale_residual:
            # one learnable scaler per residual connection
            self.residual_scalar_attn = nn.Parameter(torch.zeros(1))
            self.residual_scalar_ffn = nn.Parameter(torch.zeros(1))

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

    def forward(self, x: torch.Tensor, mask: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # forward attention (with relative position and/or temporal bias if provided)
        y = self.norm_attn(x)
        if bias is not None:
            y = self.attn(y, mask=mask, bias=bias)
        else:
            y = self.attn(y, mask=mask)
        # apply attention dropout
        y = self.dropout_attn(y)
        # residual connection: add original input back so gradients
        # can flow without having to go through attention
        if self.scale_residual:
            x = x + (self.residual_scalar_attn * y)
        else:
            x = x + y
        # forward feed forward network with norm and dropout
        y = self.norm_ffn(x)
        y = self.ffn(y)
        y = self.dropout_ffn(y)
        # residual connections
        if self.scale_residual:
            x = x + (self.residual_scalar_ffn * y)
        else:
            x = x + y
        return x


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
        scale_residual: bool = False,
        position_bias: bool = False,
        time_bias: bool = False,
        num_relative_bias_buckets: int = 32,
    ):

        super().__init__()

        # arguments
        # Either relative bias replaces absolute position embeddings. With only the time bias,
        # the model gets no position information beyond the causal mask (Guo et al., 2024 ablation).
        self.use_position_bias = position_bias
        self.use_time_bias = time_bias
        self.use_relative_bias = position_bias or time_bias
        self.num_relative_bias_buckets = num_relative_bias_buckets

        # setup item and position embeddings tables to convert integers to dense vectors
        # +1 because item IDs are 1-indexed and index 0 is reserved for padding
        self.item_embedding_table = nn.Embedding(
            num_embeddings=num_items + 1, embedding_dim=embedding_dim, padding_idx=0
        )
        # determine bias embedding table types
        if self.use_position_bias:
            self.position_bias_embedding_table = nn.Embedding(
                num_embeddings=num_relative_bias_buckets, embedding_dim=num_heads
            )
        if self.use_time_bias:
            # Separate embedding table for temporal biases, learns how time gaps
            # between interactions affect attention. Items consumed minutes apart
            # (same session) get different bias than items months apart (behavior drift).
            self.temporal_bias_embedding_table = nn.Embedding(
                num_embeddings=num_relative_bias_buckets, embedding_dim=num_heads
            )
        if not self.use_relative_bias:
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
                    scale_residual=scale_residual,
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

    def forward(self, x: torch.Tensor, timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = x.size()
        # convert item IDs to dense vectors
        item_embeddings = self.item_embedding_table(x)  # (batch_size, seq_len, embedding_dim)

        if self.use_relative_bias:
            # calculate relative position and temporal biases to use later
            # biases stay None if neither applies (time bias only, but no timestamps given)
            biases = None
            if self.use_position_bias:
                positions = torch.arange(seq_len, device=x.device)
                relative_position_distances = positions.unsqueeze(1) - positions.unsqueeze(0)  # (seq_len, seq_len)
                relative_position_buckets = bucket_relative_positions(
                    relative_position_distances, num_buckets=self.num_relative_bias_buckets
                )
                position_biases = self.position_bias_embedding_table(
                    relative_position_buckets
                )  # (seq_len, seq_len, num_heads)
                # required permutation so attention scores are compatible
                # permute from (seq_len, seq_len, num_heads) to (1, num_heads, seq_len, seq_len)
                # to broadcast with attn_scores shape (batch_size, num_heads, seq_len, seq_len)
                biases = position_biases.permute(2, 0, 1).unsqueeze(0)  # (1, num_heads, seq_len, seq_len)
            # adds temporal timestamp biases to position biases in similar fashion
            if self.use_time_bias and timestamps is not None:
                # Compute pairwise time differences between all positions in the sequence.
                # Unlike positional distances (same for all users), temporal distances are
                # per-user since each user has different interaction timestamps.
                # Shape is (batch_size, seq, seq) not (seq, seq) — each user gets unique biases.
                time_deltas = timestamps.unsqueeze(2) - timestamps.unsqueeze(1)  # (batch_size, seq_len, seq_len)
                time_buckets = bucket_temporal_distances(time_deltas, num_buckets=self.num_relative_bias_buckets)
                temporal_biases = self.temporal_bias_embedding_table(time_buckets)
                # NOTE: temporal bias is (batch_size, num_heads, seq_len, seq_len)
                # not (1, num_heads, seq_len, seq_len) because each user has different timestamps.
                temporal_biases = temporal_biases.permute(0, 3, 1, 2)  # (batch_size, num_heads, seq_len, seq_len)
                # broadcast: (1,h,s,s) + (b,h,s,s)
                biases = temporal_biases if biases is None else biases + temporal_biases
            # input is simply the item embeddings
            x = item_embeddings
            # apply dropout
            x = self.dropout_embedding(x)
            # process through entire module list of attention blocks with relative biases
            for attn_block in self.attn_blocks:
                x = attn_block(x, mask=self.causal_mask, bias=biases)  # (batch_size, seq_len, embedding_dim)
        else:
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
