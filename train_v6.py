"""
train_v6.py - SASRec + ReZero residual scaling for depth

Adds learnable residual scaling (ReZero) to make deeper models worthwhile.
v5 showed 6 blocks barely beat 2 blocks (+1% NDCG for 2.5x compute).
ReZero initializes each block as identity, letting the model learn
how much each block should contribute to the residual stream.

Run using the command: `uv run python train_v6.py --dataset ml-20m --blocks 6 --dim 64 --epochs 50 --eval-every 10`
Be sure to specify the device you want to run on using the --device argument.

References:
    - Kang & McAuley, 2018 (arxiv: 1808.09781) - SASRec architecture
    - Bachlechner et al., 2020 (arxiv: 2003.04887) - ReZero residual scaling
    - Guo et al., 2024 (arxiv: 2412.00714) - Scaling New Frontiers, depth scaling analysis
    - Klenitskiy & Vasilev, RecSys 2023 (arxiv: 2309.07602) - Sampled Softmax loss
"""

import argparse
import math
import random
from datetime import datetime, timezone

import torch
from torch.optim import Adam

from sasrec import get_git_sha, set_seed
from sasrec.dataset import load_data, load_item_titles, sample_batch
from sasrec.evaluate import demo, evaluate
from sasrec.model import SASRec


def main():
    parser = argparse.ArgumentParser(description="nanoSASRec train v6")
    parser.add_argument("--dataset", type=str, default="ml-20m", help="Dataset directory to use")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--batch-size", type=int, default=128, help="Number of examples per batch")
    parser.add_argument("--max-len", type=int, default=200, help="Max sequence length per user")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--dim", type=int, default=256, help="Embedding dimension size")
    parser.add_argument(
        "--ffn-dim", type=int, default=None, help="feed-forward network hidden dimension size (default: embedding dim)"
    )
    parser.add_argument(
        "--ffn-activation", type=str, default="relu", choices=["relu", "gelu"], help="feed-forward network activation"
    )
    parser.add_argument("--blocks", type=int, default=8, help="Number of attention blocks")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument(
        "--scale-res",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ReZero residual scaling (disable with --no-scale-res)",
    )
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
    user_train, user_valid, user_test, num_users, num_items, item_id_map, _, _, _ = load_data(
        args.data_dir, args.dataset
    )
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
        scale_residual=args.scale_res,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    model = model.to(device)

    # create and setup Adam optimizer
    # betas=(0.9, 0.98) from original Transformer paper, PyTorch default is (0.9, 0.999)
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    batches_per_epoch = num_users // args.batch_size
    print("Estimate batches per epoch:", batches_per_epoch)

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
            input_seq, pos_seq, neg_seq, _ = sample_batch(
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
