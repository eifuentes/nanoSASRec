"""
train_v10.py - HSTU with Dense All Action generative loss

Changes the training signal: instead of each position predicting only the
next item, every position predicts ALL future items in the sequence. This
gives denser supervision per sequence without changing the model.

Inspired by PinnerFormer (Pancha, Zhai et al., KDD 2022) and used in
Meta's HSTU (Zhai et al., ICML 2024). Same leading author.

The model architecture and evaluation are unchanged. Dense All Action
affects training only: the model learns richer representations by
predicting all future items, but at inference we still use the last
position to predict the next item.

Run: `uv run python train_v10.py --device cuda`

References:
    - Pancha, Zhai et al., KDD 2022 (arxiv: 2205.04507) - PinnerFormer, Dense All Action loss
    - Zhai et al., ICML 2024 (arxiv: 2402.17152) - HSTU architecture
    - Klenitskiy & Vasilev, RecSys 2023 (arxiv: 2309.07602) - Sampled Softmax loss
"""

import argparse
import random
from datetime import datetime, timezone

import torch
from torch.optim import Adam

from sasrec import get_git_sha, set_seed
from sasrec.dataset import load_data, load_item_titles, sample_batch
from sasrec.evaluate import demo, evaluate
from sasrec.model import HSTU

# Possible datasets for training
DATASETS = ("ml-1m", "ml-20m")


def main():
    parser = argparse.ArgumentParser(description="nanoSASRec train v10")
    parser.add_argument("--dataset", type=str, default="ml-20m", choices=DATASETS, help="Dataset to use")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--batch-size", type=int, default=128, help="Number of examples per batch")
    parser.add_argument("--max-len", type=int, default=200, help="Max sequence length per user")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument(
        "--horizon", type=int, default=10, help="Max future prediction horizon for Dense All Action loss"
    )
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

    # Pre-compute future target mask for Dense All Action loss (constant across batches)
    seq_len = args.max_len - 1  # due to training setup of input to positive target
    future_target_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device), diagonal=0
    ).bool()  # (seq_len, seq_len)
    # narrows the future all action view on the next K items on the horizon
    future_target_mask &= ~torch.triu(
        torch.ones(seq_len, seq_len, device=device), diagonal=args.horizon
    ).bool()  # (seq_len, seq_len)

    # Training loop
    for epoch in range(args.epochs):
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

            # Dense All Action loss (PinnerFormer, HSTU)
            # Instead of each position predicting only its next item,
            # every position predicts ALL future items in the sequence.
            #
            # v1-v9 loss (1 target per position):
            #   position 3 (saw A,B,C) -> predicts D
            #
            # v10 loss (all future targets per position):
            #   position 3 (saw A,B,C) -> predicts D, E, F, G
            #
            # This gives more gradient signal per sequence.

            # Score every output position against every target position
            # entry [i][j] = "how well does position i predict target j?"
            dense_pos_logits = torch.matmul(
                output_embedding_seq, pos_embedding_seq.transpose(1, 2)
            )  # (batch_size, seq_len, seq_len)

            # Score every output position against the shared negatives
            neg_logits_seq = torch.matmul(
                output_embedding_seq, neg_embedding_seq.transpose(1, 2)
            )  # (batch_size, seq_len, num_negs)

            # Sampled Softmax loss, memory-efficient formulation:
            # For each (position i, target j) pair, loss = logsumexp(pos, negs) - pos
            # Since negatives are shared across all targets for a given position,
            # precompute their logsumexp once per position instead of per (i,j) pair.
            neg_lse = torch.logsumexp(neg_logits_seq, dim=-1)  # (batch_size, seq_len)
            neg_lse_expanded = neg_lse.unsqueeze(-1).expand_as(dense_pos_logits)  # no new memory
            pos_neg_logits = torch.stack(
                [dense_pos_logits, neg_lse_expanded], dim=-1
            )  # (batch_size, seq_len, seq_len, 2)
            loss = torch.logsumexp(pos_neg_logits, dim=-1) - dense_pos_logits  # (batch_size, seq_len, seq_len)

            # Masking: not every [i][j] entry is a valid training signal.
            #
            # For a 5-position sequence with pos_seq = [0, A, B, C, D]:
            #
            #   target j ->   0    A    B    C    D
            #   position 0  [ .    .    .    .    .  ]   (0 = padding, all masked)
            #   position 1  [ .    ✓    ✓    ✓    ✓  ]   (predicts A, B, C, D)
            #   position 2  [ .    .    ✓    ✓    ✓  ]   (predicts B, C, D)
            #   position 3  [ .    .    .    ✓    ✓  ]   (predicts C, D)
            #   position 4  [ .    .    .    .    ✓  ]   (predicts D only)
            #
            # Two rules: (1) j >= i (upper triangle, future targets only)
            #            (2) pos_seq[j] != 0 (target must not be padding)
            target_valid = (pos_seq != 0).unsqueeze(dim=1)  # (batch_size, 1, seq_len)
            valid_mask = future_target_mask & target_valid  # (batch_size, seq_len, seq_len)

            # Calculate loss for valid loss values
            loss = loss * valid_mask
            ssml = loss.sum() / valid_mask.sum()

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
                f"Avg loss: {avg_loss:.4f} | "
                f"HR@{args.topk}: {hr:.4f} NDCG@{args.topk}: {ndcg:.4f}"
            )
        else:
            # log loss so far
            avg_loss = epoch_loss / batches_per_epoch
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
            print(f"[{ts}] [Epoch {epoch + 1}/{args.epochs}] Avg loss: {avg_loss:.4f}")

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
