"""
hstu.py - Hierarchical Sequential Transduction Unit for nanoSASRec v8

HSTU architecture (Zhai et al., ICML 2024) replacing SASRec's softmax
attention with three key changes:
    - SiLU activation replaces softmax (unnormalized, intensity-aware attention)
    - LayerNorm after aggregation (stabilizes unscaled SiLU output)
    - U(X) pointwise gating replaces FFN (multiplicative combination of
      attended context with raw input signal, structurally similar to SwiGLU)

References:
    - Zhai et al., ICML 2024 (arxiv: 2402.17152) - HSTU architecture
    - Guo et al., 2024 (arxiv: 2412.00714) - Scaling New Frontiers
    - Raffel et al., 2020 (arxiv: 1910.10683) - T5 relative position bias
    - Bachlechner et al., 2020 (arxiv: 2003.04887) - ReZero residual scaling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sasrec import bucket_relative_positions, bucket_temporal_distances, init_weights


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, dropout_rate: float, num_heads: int = 1):

        super().__init__()

        # hyper-parameters
        self.num_heads = num_heads

        # setup the attention and gating projections
        self.query_proj = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False)
        self.key_proj = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False)
        self.value_proj = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False)
        self.gate_proj = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False)
        self.out_proj = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False)

        # setup attention dropout & layer normalization
        self.attn_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.attn_dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """SiLU Dot-Product Self-Attention.

        Args:
            x (torch.Tensor): input tensor of size(batch_size, seq_len, embedding_dim)
            mask (torch.Tensor): mask to preserve causal nature of attention scores of size(seq_len, seq_len)
            bias (torch.Tensor): attention bias of size(1, num_heads, seq_len, seq_len)

        Returns:
            torch.Tensor: attended input tensor of size(batch_size, seq_len, embedding_dim)
        """
        # input tensor x is of shape (batch_size, seq_len, embedding_dim)
        batch_size, seq_len, embedding_dim = x.size()

        # infer attention head dimensions from argument num_heads
        # NOTE: should be cleanly divisible
        head_dim = embedding_dim // self.num_heads

        # project query, key, value, and gate to calculate attention scores
        # NOTE: each linear layer contains all weights for all heads for efficency
        #       and later breaks them out via reshaping

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
        # transpose key tensor for future matrix multiply so dimensions line up with query tensor
        key = key.transpose(-2, -1)  # (batch_size, num_heads, head_dim, seq_len)

        # value linear layer transforms each position into "this is the information I pass along when attended to?"
        value = self.value_proj(x)  # (batch_size, seq_len, embedding_dim)
        value = value.view(batch_size, seq_len, self.num_heads, head_dim).transpose(
            1, 2
        )  # (batch_size, num_heads, seq_len, head_dim)

        # matrix multiply aka dot product produces scores between each position and all others showing affinity
        # this creates a square matrix where each position pair generates a score, the diagonal being with itself
        # large values in each signed direction mean position pairs are compatible or incompatible
        attn_scores = torch.matmul(query, key)  # (batch_size, num_heads, seq_len, seq_len)

        # add relative position and/or temporal bias to attention scores.
        # unlike absolute position embeddings (added to item embeddings before attention),
        # this injects position information directly into the attention score matrix.
        # this keeps content signal (Q·K similarity) separate from position signal,
        # letting the model independently learn "how similar are these items" and
        # "how far apart are they in the sequence."
        # (Raffel et al. 2020 - T5, Zhai et al. 2024 - HSTU)
        attn_scores = attn_scores + bias  # (batch_size, num_heads, seq_len, seq_len)

        # apply boolean mask to protect causality
        # aka each position not having scores associated with positions to the right of it
        causal_mask = mask[:seq_len, :seq_len]
        # masked_fill with 0.0 because silu(0) = 0 * sigmoid(0) = 0
        # masked positions contribute nothing to the weighted sum downstream
        attn_scores = attn_scores.masked_fill(causal_mask, 0.0)  # (batch_size, num_heads, seq_len, seq_len)

        # SiLU self-gates each attention score: strong positive scores pass through
        # and get amplified, weak/negative scores are suppressed toward zero.
        # Unlike softmax, weights don't sum to 1, the model expresses intensity
        # of attention, not just distribution of attention.
        attn_weights = F.silu(attn_scores)  # (batch_size, num_heads, seq_len, seq_len)
        # apply dropout to further regularize attention task
        attn_weights = self.attn_dropout(attn_weights)  # (batch_size, num_heads, seq_len, seq_len)

        # Given the attention weights, does a weighted combination of each positions "value" representation
        attn_agg = torch.matmul(attn_weights, value)  # (batch_size, num_heads, seq_len, head_dim)
        # Recombine heads by reshaping, making data layout contiguous first then condensing head_dim back to embedding_dim
        attn_agg = (
            attn_agg.transpose(1, 2).contiguous().view(batch_size, seq_len, embedding_dim)
        )  # (batch_size, seq_len, embedding_dim)

        # Normalize attention aggregation due to unscaled SiLU operation earlier to stablize training
        attn_normed = self.attn_norm(attn_agg)  # (batch_size, seq_len, embedding_dim)

        # apply gating factor on attention based on original input
        gate = self.gate_proj(x)  # (batch_size, seq_len, embedding_dim)
        # applied element wise multiplication to scale each attention position by its gate
        attn_gated = torch.multiply(attn_normed, gate)  # (batch_size, seq_len, embedding_dim)

        # Linearly combine everything to product a final attention sequence embedding representation
        return self.out_proj(attn_gated)  # (batch_size, seq_len, embedding_dim)


class HSTULayer(nn.Module):
    def __init__(
        self,
        dropout_rate: float,
        embedding_dim: int,
        num_heads: int = 1,
    ):

        super().__init__()

        # scale residual ReZero parameters (Bachlechner et al. 2020)
        # one learnable scalar per residual connection
        self.residual_scalar_attn = nn.Parameter(torch.zeros(1))
        # attention module
        self.attn = SelfAttention(embedding_dim=embedding_dim, dropout_rate=dropout_rate, num_heads=num_heads)
        # dropout module
        self.dropout_attn = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        # forward attention (with relative position and/or temporal bias)
        y = self.attn(x, mask=mask, bias=bias)
        # apply attention dropout
        y = self.dropout_attn(y)
        # residual connection: add original input back so gradients
        # can flow without having to go through attention
        # scale the attention path so as to now overwhelm initial input
        y = x + (self.residual_scalar_attn * y)
        return y


class HSTU(nn.Module):
    def __init__(
        self,
        num_items: int,
        max_seq_len: int,
        dropout_rate: float,
        embedding_dim: int,
        num_blocks: int = 2,
        num_heads: int = 1,
        num_relative_bias_buckets: int = 32,
    ):

        super().__init__()

        # hyper-parameters
        self.num_relative_bias_buckets = num_relative_bias_buckets

        # setup item and position embeddings tables to convert integers to dense vectors
        # +1 because item IDs are 1-indexed and index 0 is reserved for padding
        self.item_embedding_table = nn.Embedding(
            num_embeddings=num_items + 1, embedding_dim=embedding_dim, padding_idx=0
        )
        # setup bias embedding table by type
        # relative position bias embedding table
        self.position_bias_embedding_table = nn.Embedding(
            num_embeddings=num_relative_bias_buckets, embedding_dim=num_heads
        )
        # relative temporal bias embedding table
        # learns how time gaps between interactions affect attention.
        # Items consumed minutes apart (same session) get different bias
        # than items months apart (behavior drift).
        self.temporal_bias_embedding_table = nn.Embedding(
            num_embeddings=num_relative_bias_buckets, embedding_dim=num_heads
        )
        # dropout layer
        self.dropout_embedding = nn.Dropout(p=dropout_rate)
        # final norm layer
        self.final_norm = nn.LayerNorm(embedding_dim)
        # Build attention blocks
        self.hstu_layers = nn.ModuleList(
            [
                HSTULayer(
                    dropout_rate=dropout_rate,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                )
                for _ in range(num_blocks)
            ]
        )
        # Apply truncated normal weight initialization
        self.apply(lambda m: init_weights(m, embedding_dim))
        # create boolean mask to protect causality in forward pass
        # aka each position not having scores associated with positions to the right of it
        # NOTE: do not need to add other 2 dimensions to `ones` creation due to broadcasting in masked_fill later
        mask = torch.triu(torch.ones((max_seq_len - 1, max_seq_len - 1)), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.size()

        # calculate relative position biases to use later
        positions = torch.arange(seq_len, device=x.device)
        relative_position_distances = positions.unsqueeze(0) - positions.unsqueeze(1)  # (seq_len, seq_len)
        relative_position_buckets = bucket_relative_positions(
            relative_position_distances, num_buckets=self.num_relative_bias_buckets
        )
        position_biases = self.position_bias_embedding_table(
            relative_position_buckets
        )  # (seq_len, seq_len, num_heads)
        # required permutation so attention scores are compatible
        # permute from (seq_len, seq_len, num_heads) to (1, num_heads, seq_len, seq_len)
        # to broadcast with attn_scores shape (batch_size, num_heads, seq_len, seq_len)
        position_biases = position_biases.permute(2, 0, 1).unsqueeze(0)  # (1, num_heads, seq_len, seq_len)

        # adds temporal timestamp biases to position biases in similar fashion
        # Compute pairwise time differences between all positions in the sequence.
        # Unlike positional distances (same for all users), temporal distances are
        # per-user since each user has different interaction timestamps.
        # Shape is (batch_size, seq, seq) not (seq, seq), each user gets unique biases.
        time_deltas = timestamps.unsqueeze(1) - timestamps.unsqueeze(2)  # (batch_size, seq_len, seq_len)
        time_buckets = bucket_temporal_distances(time_deltas, num_buckets=self.num_relative_bias_buckets)
        temporal_biases = self.temporal_bias_embedding_table(time_buckets)
        # NOTE: temporal bias is (batch_size, num_heads, seq_len, seq_len)
        # not (1, num_heads, seq_len, seq_len) because each user has different timestamps.
        temporal_biases = temporal_biases.permute(0, 3, 1, 2)  # (batch_size, num_heads, seq_len, seq_len)

        # combine position and temporal biases into all attention modules
        biases = position_biases + temporal_biases  # broadcast: (1,h,s,s) + (b,h,s,s)

        # input is simply the item embeddings
        # convert item IDs to dense vectors
        x = self.item_embedding_table(x)  # (batch_size, seq_len, embedding_dim)
        # apply dropout
        x = self.dropout_embedding(x)  # (batch_size, seq_len, embedding_dim)
        # process through entire module list of attention blocks with computed biases and pre-computed causal mask
        for hstu_layer in self.hstu_layers:
            x = hstu_layer(x, mask=self.causal_mask, bias=biases)  # (batch_size, seq_len, embedding_dim)

        # final layer normalization since attention ends in residual add
        return self.final_norm(x)  # (batch_size, seq_len, embedding_dim)
