# nanoSASRec

The simplest, most readable implementation of Self-Attentive Sequential Recommendation (SASRec), following the research line that made it the standard baseline and the improvements that eventually led to the Hierarchical Sequential Transduction Unit (HSTU).

The official curriculum is 10 versions: nine training scripts plus the v5 depth sweep. v1-v4 are single files you can read top to bottom; v6+ share a small `sasrec/` package.

Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT): every training version is runnable directly, every design decision cites a paper, and every version has a concrete reproduction target.

## Why this exists

Most recommender systems (collaborative filtering, two-tower models, deep learning recommender models) learn what a user *likes*. They struggle with what a user *needs right now*. On Amazon, someone who just bought a DSLR camera doesn't need more camera recommendations. They need a lens, then a memory card, then a camera bag. With movies, a user who just watched all three Lord of the Rings films back-to-back is on a fantasy binge, not looking for the romantic comedies their five-year profile suggests. The sequence encodes intent progression that a static user profile can't capture.

This matters because real user behavior is *streaky*. People shop in projects, watch in moods, and listen in phases. Sequential models pick up on these short-term patterns while still remembering long-term preferences, which is why they have become the standard in production recommendation at companies like Meta, YouTube and Kuaishou.

Sequential recommendation models the *order* of a user's actions. GRU4Rec (2016) brought RNNs to it, and Transformers turned out to be better at it. **SASRec** (Kang & McAuley, 2018) was the first to apply a causal Transformer decoder to item sequences.

This project builds up SASRec from its original framing all the way to its HSTU variant, one component at a time. Each step makes a change, measures what it does, and cites the paper behind it. To use LLMs as an analogy: if SASRec is the original Transformer, HSTU is GPT-2. Same model lineage, but with the refinements that unlocked scaling laws. By v10 you'll understand causal self-attention, loss design for large catalogs, how depth scales, and which production techniques do and don't transfer to a public benchmark.

## Quick start

```bash
uv sync
uv run python data.py --dataset ml-1m
uv run python train_v1.py --epochs 200 --eval-every 20 --device mps  # or cpu, cuda
```

Train v1 on MovieLens 1M in about 8 minutes on a MacBook GPU (`mps`). The [Versions](#versions) table below shows how the curriculum progresses, and [CURRICULUM.md](CURRICULUM.md) walks through every version: the command, the output, the numbers, and what works, what doesn't, and why. The 8-block ML-20M versions take about 3 to 4 hours for 50 epochs on one A10 GPU.

Needs Python 3.12+ and PyTorch 2.7+. No frameworks, no config systems, no custom CUDA kernels.

## What it looks like

```
Recent history:  Batman (1989), Batman Returns (1992), Batman & Robin (1997)
Ground truth:    Batman Forever (1995)
Model top-3:
  1. Batman Forever (1995)
  2. Superman III (1983)
  3. Avengers, The (1998)
```

A hand-picked test user, scored by v4 on ML-1M. The model never trains on the ground truth, and it ranks it first among every movie the user hasn't rated yet. Most users aren't this easy: v4 puts the ground truth in the top 10 for about 30% of them.

## Versions

| Version | What changes | Dataset | Key finding |
|------|-------------|---------|-------------|
| `train_v1.py` | Vanilla SASRec + BCE loss | ML-1M | BCE overconfidence problem |
| `train_v2.py` | Sampled softmax loss | ML-1M | Competition between candidates fixes ranking |
| `train_v3.py` | gBCE loss + calibration | ML-1M | Calibration needs 2× the epochs; sampled softmax still ranks better |
| `train_v4.py` | Init, reversed positions, dropout, full-catalog eval, block-wiring flags | ML-1M | Implementation details add +12% NDCG on sampled validation |
| `train_v5.sh` | Depth sweep (2/4/8/16 blocks) | ML-20M | Depth pays off with diminishing returns, but only with pre-LN blocks; a widely copied wiring fails to train from 8 blocks |
| `train_v6.py` | Gated residuals (ReZero) | ML-20M | 8 blocks with ReZero match 16 blocks without it |
| `train_v7.py` | Relative position + temporal bias | ML-20M | Best model in the repo on ML-20M |
| `train_v8.py` | SiLU + gating (HSTU) | ML-20M | Not a clear win at ML-20M's scale |
| `train_v9.py` | φ1 SwishLayerNorm pre-activation | ML-20M | Recovers most of v8's gap to v7 |
| `train_v10.py` | Temperature, L2 norm, collision masking | ML-20M | Didn't help in our setup |

Each version is runnable independently. Read v1-v10 in order for the full progression, or jump to any version. The terms in the table are defined in [CURRICULUM.md](CURRICULUM.md#terms).

`train_v11.py` is an appendix experiment for Dense All Action loss. It is useful context, but not part of the official curriculum path.

## Datasets

| Dataset | Role | Users | Items | Avg len | When used |
|---------|------|-------|-------|---------|-----------|
| ML-1M | Curriculum baseline | 6,040 | 3,416 | 165.5 | v1-v4 |
| ML-20M | Benchmark target | 138,493 | 26,744 | 144.4 | v5-v10 |

- **ML-1M** follows the SASRec (Kang & McAuley 2018) and BERT4Rec (Petrov & Macdonald 2022) protocol: 5-core filtering, which repeatedly drops users and items with fewer than 5 interactions, leaving 3,416 of the 3,706 movies.
- **ML-20M** follows the HSTU (Zhai et al. 2024) and eSASRec (Tikhonovich et al. 2025) protocol: no filtering, so the full 26,744-item catalog is kept.

In both, each user's history is sorted by time, and interactions with tied timestamps keep the order of the source file.

## Evaluation

`data.py --dataset <name>` writes the split to `data/<name>/split.json`: for each user, the training sequence, the validation item and the test item, with timestamps, plus a map from item index back to the MovieLens movie ID. Every training script reads it, so all versions train and evaluate on the same split. A `metadata.json` next to it records the source file, its MD5 and the protocol. `data.py` checks each download against the MD5 GroupLens publishes, and the ML-20M counts against the published numbers.

Each history is split leave-one-out: the last item is the test target, the one before it the validation target, and the rest is training data ([diagram](CURRICULUM.md#reading-the-metrics)). From v4 onward, validation and test rank the ground truth against the full item catalog, leaving out items already in the user's history. v1-v3 keep sampled evaluation (the ground truth against 100 random items), because they teach loss functions.

## Benchmarks

Full-catalog evaluation on ML-20M (dim=256, 8 heads, one A10 GPU): every model ranks the ground truth against all 26,744 items, not a sampled subset. HR@10 and NDCG@10 are explained in [Reading the Metrics](CURRICULUM.md#reading-the-metrics).

**Compare rows at the same number of epochs.** Every version was run for 50 epochs and was still improving there. Five were also run for 100 epochs, the budget eSASRec uses. Running every version at 100 epochs and 3 seeds is [#25](https://github.com/eifuentes/nanoSASRec/issues/25).

| Version | What changed | Epochs | HR@10 | NDCG@10 |
|---------|-------------|--------|-------|---------|
| v5 (2 blocks) | SASRec baseline | 50 | 0.2938 | 0.1713 |
| v5 (4 blocks) | + depth | 50 | 0.3098 | 0.1830 |
| v5 (8 blocks) | + more depth | 50 | 0.3196 | 0.1912 |
| v5 (8 blocks) | + more depth | 100 | 0.3297 | 0.1976 |
| v5 (16 blocks) | + even more depth | 50 | 0.3280 | 0.1970 |
| v6 (8 blocks) | + ReZero | 50 | 0.3287 | 0.1973 |
| v7 (8 blocks) | + relative bias | 50 | 0.3380 ± 0.0009 | 0.2025 ± 0.0011 |
| **v7 (8 blocks)** | **+ relative bias** | **100** | **0.3473 ± 0.0013** | **0.2098 ± 0.0005** |
| v8 (8 blocks) | HSTU architecture | 50 | 0.3236 | 0.1929 |
| v8 (8 blocks) | HSTU architecture | 100 | 0.3313 | 0.1976 |
| v9 (8 blocks) | + φ1 pre-activation | 50 | 0.3354 ± 0.0009 | 0.2008 ± <0.0001 |
| v9 (8 blocks) | + φ1 pre-activation | 100 | 0.3426 | 0.2061 |
| v10 (8 blocks) | + temp + L2 + collision | 50 | 0.3269 | 0.1949 |
| v10 (8 blocks) | + temp + L2 + collision | 100 | 0.3336 | 0.1992 |

The ± rows are the mean and standard deviation over seeds 42, 43 and 44 (`--seed`); the others use seed 42. Seed-to-seed spread is about 0.1 points of HR@10, well below the gaps between versions. The order of the versions is the same at 50 and 100 epochs. [CURRICULUM v8](CURRICULUM.md#v8-hstu) looks at how each version gains from the extra epochs.

Published results on the full 26,744-item ML-20M catalog:

| Method | Split | Numbers from | HR@10 | NDCG@10 |
|--------|-------|--------------|-------|---------|
| **v7 (8 blocks, 100 epochs)** | same as ours | [this repo](#benchmarks) | **0.3473 ± 0.0013** | **0.2098 ± 0.0005** |
| eSASRec (4 blocks) | same as ours | [eSASRec paper](PAPERS.md#evaluation-methodology) | 0.3289 | 0.1966 |
| eSASRec (8 blocks) | same as ours | [eSASRec paper](PAPERS.md#evaluation-methodology) | 0.3457 | 0.2107 |
| HSTU (8 blocks) | same as ours | [eSASRec paper, re-run](PAPERS.md#evaluation-methodology) | 0.3464 | 0.2040 |
| FuXi-α (8 blocks) | same as ours | [eSASRec paper, re-run](PAPERS.md#evaluation-methodology) | 0.3468 | 0.2043 |
| SASRec | HSTU's | [HSTU paper](PAPERS.md#the-bridge-to-hstu-2022-2024) | 0.2889 | 0.1621 |
| HSTU (4 blocks) | HSTU's | [HSTU paper](PAPERS.md#the-bridge-to-hstu-2022-2024) | 0.3273 | 0.1895 |
| HSTU-large (16 blocks) | HSTU's | [HSTU paper](PAPERS.md#the-bridge-to-hstu-2022-2024) | 0.3556 | 0.2098 |

The rows on our split are the closest comparison: same split, same full-catalog ranking. HSTU's split orders tied timestamps differently, which changes about 25% of the test targets, so those rows aren't directly comparable.

At 100 epochs, v7 ties eSASRec and the re-run HSTU and FuXi-α. Its HR@10 is 0.1 to 0.2 points higher, about one standard deviation across seeds, so it's a tie, not a win. Our best HSTU, v9, stays 0.4 points of HR@10 behind the re-run HSTU. Our HSTU still differs from Meta's code in a few places, and [#30](https://github.com/eifuentes/nanoSASRec/issues/30) tests which ones matter.

### Reproducing these numbers

Every ML-20M script starts from the same recipe:

| Setting | Value |
|---------|-------|
| Embedding size / blocks / heads | 256 / 8 / 8 (`--dim`, `--blocks`, `--heads`) |
| Sequence length / batch size | 200 / 128 (`--max-len`, `--batch-size`) |
| Loss | Sampled softmax, 512 random negatives per user (`--num-negatives`) |
| Optimizer | Adam, learning rate 1e-3 (constant), betas (0.9, 0.98) |
| Dropout | 0.2 |
| Budget | 50 epochs, seed 42 (`--epochs`, `--seed`) |
| v7 adds | ReZero (`--residual-gate rezero`) and relative position + time bias (`--rel-bias`), both on by default |

Each row above is one command (add `--device cuda`, or run it on Modal with `remote.py`):

| Row | Command |
|-----|---------|
| v5, all four depths | `bash train_v5.sh cuda` (also runs v5's block-wiring comparison) |
| v5, one depth | `uv run python train_v4.py --dataset ml-20m --dim 256 --heads 8 --num-negatives 512 --epochs 50 --eval-every 10 --blocks 8` |
| v6 to v10 | `uv run python train_v6.py`, and the same for `train_v7.py` to `train_v10.py` |
| v7 and v9, three seeds | add `--seed 42`, `--seed 43` and `--seed 44` |
| 100 epochs | add `--epochs 100`. A finished 50-epoch run resumes from its checkpoint instead of starting over. |
| v7 with eSASRec's settings ([result](CURRICULUM.md#v7-relative-position--temporal-bias)) | `uv run python train_v7.py --ffn-dim 1024 --heads 16 --num-negatives 256 --epochs 100` |

See [CURRICULUM.md](CURRICULUM.md) for the full analysis from v1 to v10.

## Papers

See [PAPERS.md](PAPERS.md) for the full research lineage from GRU4Rec through HSTU and which paper justifies which code decision.

## Remote GPU Training

For longer ML-20M runs, `remote.py` can launch any `train_v*.py` script on [Modal](https://modal.com/). Create a Modal account and authenticate locally before using it.

```bash
uv sync --group cloud
uv run modal run --detach remote.py --cmd "python -u train_v9.py --dataset ml-20m"
```

You can also use a regular GPU VM from providers like [Lambda](https://lambda.ai/instances), [Lightning AI](https://lightning.ai/), or any cloud where `uv sync` can run. This project is not sponsored. [DETAILS.md](DETAILS.md#remote-training-on-modal) has tips for running several Modal jobs at once.

### Checkpoints

v4+ scripts save a checkpoint every 5 epochs (`--checkpoint-every`) and after the last epoch, under `checkpoints/`. After any interruption, run the same command again and it picks up where it left off. To train a finished run for longer, run it again with a larger `--epochs`. [DETAILS.md](DETAILS.md#checkpoints) covers how runs are named, when a resumed run matches an uninterrupted one exactly, and what to do after changing code.

## Contributing

Install the dev tools, then run Ruff and the tests before committing:

```bash
uv sync --group dev
uv run ruff format .
uv run ruff check .
uv run pytest
```

The tests run on CPU in a few seconds on tiny synthetic data, and GitHub Actions runs the same commands on every pull request. They cover the evaluation protocol, causal masking in every model, how training batches are built, and exact resume from checkpoints.

What's next is tracked in [GitHub issues](https://github.com/eifuentes/nanoSASRec/issues): pick one up, or open a new one.
