# pyright: reportArgumentType=false
"""
train_v8.py — HSTU: Hierarchical Sequential Transduction Unit

Single-file implementation of HSTU (Zhai et al., ICML 2024) replacing
SASRec's softmax attention with three key changes:
    - SiLU activation replaces softmax (unnormalized, intensity-aware attention)
    - LayerNorm after aggregation (stabilizes unscaled SiLU output)
    - U(X) pointwise gating replaces FFN (multiplicative combination of
      attended context with raw input signal, structurally similar to SwiGLU)

Also includes relative position + temporal bias (v7), ReZero residual
scaling (v6), and numerically stable logsumexp loss formulation.

Run: `uv run python train_v8.py --device cuda`

References:
    - Zhai et al., ICML 2024 (arxiv: 2402.17152) - HSTU architecture
    - Guo et al., 2024 (arxiv: 2412.00714) - Scaling New Frontiers
    - Raffel et al., 2020 (arxiv: 1910.10683) - T5 relative position bias
    - Bachlechner et al., 2020 (arxiv: 2003.04887) - ReZero residual scaling
    - Klenitskiy & Vasilev, RecSys 2023 (arxiv: 2309.07602) - Sampled Softmax loss
"""

import argparse
import math
import random
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from sasrec import checkpoint, get_git_sha, set_seed
from sasrec.dataset import load_data, load_item_titles, sample_batch
from sasrec.evaluate import demo, evaluate
from sasrec.model.sasrec import (
    NUM_BIAS_BUCKETS,
    bucket_relative_positions,
    bucket_temporal_distances,
    init_weights,
)

# Possible datasets for training
DATASETS = ("ml-1m", "ml-20m")


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
        # NOTE: each linear layer contains all weights for all heads for efficiency
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

        # Normalize attention aggregation due to unscaled SiLU operation earlier to stabilize training
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
        # scale the attention path so as not to overwhelm initial input
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
        num_relative_bias_buckets: int = NUM_BIAS_BUCKETS,
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
        # Bucket the distance between every query and key position once, here, rather than
        # in every forward pass. Distance i - j counts how many steps back key j is from query i.
        # persistent=False keeps it out of saved checkpoints, since it can always be rebuilt.
        positions = torch.arange(max_seq_len - 1)
        distances = positions.unsqueeze(1) - positions.unsqueeze(0)  # (max_seq_len - 1, max_seq_len - 1)
        # the farthest a key can be from its query is max_seq_len - 2 steps
        buckets = bucket_relative_positions(distances, max_seq_len - 2, num_relative_bias_buckets)
        self.register_buffer("position_buckets", buckets, persistent=False)

    def forward(self, x: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.size()

        # calculate relative position biases to use later
        # every (query, key) distance was bucketed once in __init__, so just slice
        relative_position_buckets = self.position_buckets[:seq_len, :seq_len]  # (seq_len, seq_len)
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
        time_deltas = timestamps.unsqueeze(2) - timestamps.unsqueeze(1)  # (batch_size, seq_len, seq_len)
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


def main():
    parser = argparse.ArgumentParser(description="nanoSASRec train v8")
    parser.add_argument("--dataset", type=str, default="ml-20m", choices=DATASETS, help="Dataset to use")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--batch-size", type=int, default=128, help="Number of examples per batch")
    parser.add_argument("--max-len", type=int, default=200, help="Max sequence length per user")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--dim", type=int, default=256, help="Embedding dimension size")
    parser.add_argument("--blocks", type=int, default=8, help="Number of attention blocks")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for optimizer during training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training")
    parser.add_argument("--num-negatives", type=int, default=512, help="Number of negatives to sample for training")
    parser.add_argument("--topk", type=int, default=10, help="Cutoff for HR@K and NDCG@K")
    parser.add_argument("--eval-every", type=int, default=10, help="Evaluate every N epochs")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run computation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Exact run-to-run results on the same GPU model, about 20%% slower (see README)",
    )
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
    set_seed(args.seed, deterministic=args.deterministic)

    # Determine device
    device = torch.device(args.device)
    if torch.cuda.is_available() and args.device == "cpu":
        print("WARNING: CUDA is available but running on CPU. Use --device cuda for faster training.")

    # Load pre-processed dataset with train, validation, test splits
    (
        user_train,
        user_valid,
        user_test,
        num_users,
        num_items,
        item_id_map,
        user_train_ts,
        user_valid_ts,
        _,
    ) = load_data(args.data_dir, args.dataset)
    print("Dataset Lengths:", len(user_train), len(user_valid), len(user_test))
    # load titles for future demos
    titles = load_item_titles(args.data_dir, args.dataset, item_id_map)

    # Init HSTU model with hyperparams
    model = HSTU(
        num_items=num_items,
        max_seq_len=args.max_len,
        dropout_rate=args.dropout,
        embedding_dim=args.dim,
        num_blocks=args.blocks,
        num_heads=args.heads,
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
    run_checkpoint_dir = checkpoint.run_dir(args.checkpoint_dir, parser.prog, args)
    start_epoch = checkpoint.resume(run_checkpoint_dir, model, optimizer)

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
            input_seq, pos_seq, neg_seq, input_timestamps = sample_batch(
                user_train,
                num_users=num_users,
                num_items=num_items,
                max_len=args.max_len,
                batch_size=args.batch_size,
                num_negs=args.num_negatives,
                timestamps=user_train_ts,
            )

            # Convert numpy int64 arrays to torch LongTensors
            input_seq = torch.as_tensor(input_seq).to(device)  # (batch_size, seq_len)
            pos_seq = torch.as_tensor(pos_seq).to(device)  # (batch_size, seq_len)
            neg_seq = torch.as_tensor(neg_seq).to(device)  # (batch_size, num_negs)
            assert input_timestamps is not None, "HSTU requires timestamps - re-run data.py to generate them"
            input_timestamps = torch.as_tensor(input_timestamps).to(device)  # (batch_size, seq_len)

            # Determine embeddings for input, positive and negative sequences
            output_embedding_seq = model.forward(
                input_seq, timestamps=input_timestamps
            )  # (batch_size, seq_len, embedding_dim)
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
            # Stable logsumexp form of the same sampled-softmax objective.
            # stack positive and negatives together: (batch_size, seq_len, 1 + num_negs)
            logits_seq = torch.cat([pos_logits_seq.unsqueeze(dim=-1), neg_logits_seq], dim=-1)
            # logsumexp across candidates, then subtract the positive score
            loss = torch.logsumexp(logits_seq, dim=-1) - pos_logits_seq

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
                    user_train_ts=user_train_ts,
                    user_valid_ts=user_valid_ts,
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
            checkpoint.save(run_checkpoint_dir, epoch + 1, model, optimizer, args)

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
            user_train_ts=user_train_ts,
            user_valid_ts=user_valid_ts,
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
            demo(
                demo_user,
                model,
                user_train,
                user_valid,
                user_test,
                num_items,
                args.max_len,
                device,
                titles,
                user_train_ts=user_train_ts,
                user_valid_ts=user_valid_ts,
            )


if __name__ == "__main__":
    main()
