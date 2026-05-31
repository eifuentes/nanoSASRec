"""
model.py - SASRec model architecture for nanoSASRec v4+

Shared model components used by train_v4.py and later versions.
Architecture is identical to v1-v3 with these refinements:
    - Weight init: truncated normal with std=1/sqrt(d) (Meta's SASRec)
    - Reversed positional embeddings (Huang 2025, arxiv: 2504.09596)
    - Causal mask as registered buffer (created once, not per forward pass)
    - Configurable feed-forward network hidden dim and activation function
    - Attention weight dropout (Vaswani et al. 2017, Section 3.1)
    - Residual scaling via ReZero (Bachlechner et al. 2020) for stable depth scaling
"""

import math

import torch
import torch.nn as nn
from typing import Optional, Literal


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

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Scaled Dot-Product Self-Attention.

        Args:
            x (torch.Tensor): input tensor of size(batch_size, seq_len, embedding_dim)
            mask (torch.Tensor): mask to preserve causal nature of attention scores

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

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # forward attention
        y = self.norm_attn(x)
        y = self.attn(y, mask=mask)
        y = self.dropout_attn(y)
        # residual connection: add original input back so gradients
        # can flow without having to go through attention
        if self.scale_residual:
            x = x + (self.residual_scalar_attn * y)
        else:
            x = x + y
        # forward feed forward network
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
        embeddings = item_embeddings + position_embeddings  # automatic broadcast
        # apply dropout
        embeddings = self.dropout_embedding(embeddings)
        # process through entire module list of attention blocks
        x = embeddings
        for attn_block in self.attn_blocks:
            x = attn_block(x, mask=self.causal_mask)  # (batch_size, seq_len, embedding_dim)
        # final layer normalization since attention ends in residual add
        return self.final_norm(x)  # (batch_size, seq_len, embedding_dim)
