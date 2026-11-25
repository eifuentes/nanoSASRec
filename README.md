# nanoSASRec

The simplest, most readable implementation of Self-Attentive Sequential Recommendation (SASRec), following the research line that made it the standard baseline and the improvements that eventually led to the Hierarchical Sequential Transduction Unit (HSTU).

The official curriculum is 10 training scripts backed by a small shared `sasrec/` package.

Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT): every training version is runnable directly, every design decision cites a paper, and every version has a concrete reproduction target.

## Why this exists

Most recommender systems (collaborative filtering, two-tower models, deep learning recommender models) learn what a user *likes*. They struggle with what a user *needs right now*. On Amazon, someone who just bought a DSLR camera doesn't need more camera recommendations. They need a lens, then a memory card, then a camera bag. With movies, a user who just watched all three Lord of the Rings films back-to-back is on a fantasy binge, not looking for the romantic comedies their five-year profile suggests. The sequence encodes intent progression that a static user profile can't capture.

This matters because real user behavior is *streaky*. People shop in projects, watch in moods, and listen in phases. Sequential models pick up on these short-term patterns while still remembering long-term preferences, which is why they recently dominate production recommendation at companies like Meta, YouTube, Kuaishou, and Meituan.

Sequential recommendation models the *order* of a user's actions. RNNs (GRU4Rec, 2016) did it first, but Transformers turned out to be better at it. **SASRec** (Kang & McAuley, 2018) was the first to apply a causal Transformer decoder to item sequences.

This project builds up SASRec from its original framing all the way to its HSTU variant, one component at a time. Each step fixes one problem, shows the metric improvement, and cites the paper that figured it out. To use LLMs as an analogy: if SASRec is the original Transformer, HSTU is GPT-2. Same model lineage, but with the refinements that unlocked scaling laws. By v10 you'll understand causal self-attention, loss function design for large catalogs, scaling laws for recommendation, and the specific architectural decisions that separate a research baseline from a production system.

## Quick start

```bash
uv sync
uv run python data.py --dataset ml-1m
uv run python train_v1.py --epochs 200 --eval-every 20  # --device cuda
```

Train v1 on MovieLens 1M in ~7 minutes on a MacBook. From there, use the [Versions](#versions) and [Datasets](#datasets) tables to see how the curriculum progresses, then follow [CURRICULUM.md](CURRICULUM.md) for the v1-v10 commands and expected results. Reproduce the Zhai et al. (2024) HSTU MovieLens 20M benchmark in ~4 hours on a single A100.

## What to expect

See [CURRICULUM.md](CURRICULUM.md) for a walkthrough of the 10 official versions with training outputs, benchmark numbers, and detailed analysis of what works, what doesn't, and why.

## What it looks like

```
Recent history:  Toy Story (1995), Bug's Life (1998), Antz (1998)
Ground truth:    Pocahontas (1995)
Model top-3:
  1. Lion King, The (1994)
  2. Little Mermaid, The (1989)
  3. Jungle Book, The (1967)
```

## Versions

| Version | What changes | Dataset | Key finding |
|------|-------------|---------|-------------|
| `train_v1.py` | Vanilla SASRec + BCE loss | ML-1M | BCE overconfidence problem |
| `train_v2.py` | Sampled softmax loss | ML-1M | Competition between candidates fixes ranking |
| `train_v3.py` | gBCE loss + calibration | ML-1M | Calibrated probabilities, but SSM ranks better |
| `train_v4.py` | Init, reversed positions, dropout | ML-1M | Implementation details compound to +14% NDCG |
| `train_v5.sh` | Depth sweep (2/4/8/16 blocks) | ML-20M | Depth pays off, diminishing returns after 8 |
| `train_v6.py` | ReZero residual scaling | ML-20M | 8 blocks + ReZero matches 16 blocks without |
| `train_v7.py` | Relative position + temporal bias | ML-20M | Best model in repo |
| `train_v8.py` | SiLU + gating (HSTU) | ML-20M | Architecture swap is not a clear win at this scale |
| `train_v9.py` | φ1 SwishLayerNorm pre-activation | ML-20M | Recovers most of v8's gap to v7 |
| `train_v10.py` | Temperature, L2 norm, collision masking | ML-20M | Production techniques need in-batch negatives to help |

Each version is runnable independently. Read v1-v10 in order for the full progression, or jump to any version.

`train_v11.py` is an appendix experiment for Dense All Action loss. It is useful context, but not part of the official curriculum path.

## Datasets

| Dataset | Role | Users | Items | Avg len | When used |
|---------|------|-------|-------|---------|-----------|
| ML-1M | Curriculum baseline | 6,040 | 3,416 | 165.5 | v1-v4 |
| ML-20M | Benchmark target | 138,493 | 26,744 | 144.4 | v5-v10 |

ML-20M follows the HSTU (Zhai et al. 2024) and eSASRec (Tikhonovich et al. 2025) protocol. There is no k-core filtering, so the full 26,744-item catalog is kept, and interactions with tied timestamps keep their `ratings.csv` order. `data.py` verifies each download against the MD5 GroupLens publishes, checks the processed statistics against the published numbers, and writes a `metadata.json` next to `split.json` that records the source and protocol. Up to v0.2.1, nanoSASRec dropped items with fewer than 5 ratings (18,345 items). That made ranking easier and made the numbers not comparable with published ML-20M results.

From v4 onward, validation and test metrics rank the ground truth against the full item catalog, excluding items already in the user's history. v1-v3 keep sampled validation (100 negatives) because they teach loss functions.

## Benchmarks

Full-catalog evaluation on ML-20M (dim=256, 8 heads, 50 epochs, one A10 GPU). Every model ranks the ground truth against all 26,744 items, not a sampled subset. v7 and v9 are the mean ± standard deviation over seeds 42, 43 and 44. The other rows use seed 42.

| Version | What changed | HR@10 | NDCG@10 |
|---------|-------------|-------|---------|
| v5 (2 blocks) | SASRec baseline | 0.2938 | 0.1713 |
| v5 (4 blocks) | + depth | 0.3098 | 0.1830 |
| v5 (8 blocks) | + more depth | 0.3196 | 0.1912 |
| v5 (16 blocks) | + even more depth | 0.3280 | 0.1970 |
| v6 (8 blocks) | + ReZero | 0.3287 | 0.1973 |
| **v7 (8 blocks)** | **+ relative bias** | **0.3380 ± 0.0009** | **0.2025 ± 0.0011** |
| v8 (8 blocks) | HSTU architecture | 0.3236 | 0.1929 |
| v9 (8 blocks) | + φ1 pre-activation | 0.3354 ± 0.0009 | 0.2008 ± 0.0000 |
| v10 (8 blocks) | + temp + L2 + collision | 0.3269 | 0.1949 |

Seed-to-seed spread is about 0.1 points of HR@10, well below the gaps between versions. v0.2.1 reported these versions on the item 5-core ML-20M (18,345 items). Those numbers were 0.4 to 1.0 points of HR@10 higher, because the smaller catalog makes ranking easier, but the 8-block versions ranked in the same order (v7 > v9 > v6 > v10 > v8 > v5).

Published results on the full 26,744-item ML-20M catalog:

| Source | Method | HR@10 | NDCG@10 |
|--------|--------|-------|---------|
| Zhai et al. (2024) | SASRec | 0.2889 | 0.1621 |
| Zhai et al. (2024) | HSTU | 0.3273 | 0.1895 |
| Zhai et al. (2024) | HSTU-large | 0.3556 | 0.2098 |
| Tikhonovich et al. (2025) | eSASRec (4 blocks) | 0.3289 | 0.1966 |
| Tikhonovich et al. (2025) | eSASRec (8 blocks) | 0.3457 | 0.2107 |
| Tikhonovich et al. (2025) | HSTU, recomputed (8 blocks) | 0.3464 | 0.2040 |

eSASRec's numbers come from the [transformer_benchmark](https://github.com/blondered/transformer_benchmark) README. Its HSTU row is HSTU's original code re-run on the same order-preserving split that `data.py` produces.

The eSASRec rows are the closest comparison because they use the same split. At 8 blocks, v7 is 0.8 points of HR@10 below eSASRec and the recomputed HSTU. Our HSTU versions trail the recomputed HSTU by more: 1.1 points for v9 and 2.3 for v8. One known difference is that nanoSASRec does not yet handle long sequences the way the reference training does (see Stochastic Length training in the [Roadmap](#roadmap)). Zhai et al.'s own rows come from preprocessing that reorders tied timestamps, so about 25% of their test interactions differ from ours.

See [CURRICULUM.md](CURRICULUM.md) for the full analysis from v1 to v10.

## Papers

See [PAPERS.md](PAPERS.md) for the full research lineage from GRU4Rec through HSTU and which paper justifies which code decision.

## Requirements

Python 3.12+ and PyTorch 2.7+. No frameworks, no config systems, no custom CUDA kernels.

```toml
dependencies = [
    "torch>=2.7",
    "numpy>=1.24",
]
```

If you plan to edit the repo, install the dev tools and run Ruff and the tests before committing changes:

```bash
uv sync --group dev
uv run ruff format .
uv run ruff check .
uv run pytest
```

The tests are deliberately few. They cover the protocol decisions that published comparisons depend on: the leave-one-out split with its tie-break, and full-catalog ranking with history masking.

## Remote GPU Training

For longer ML-20M runs, `remote.py` can launch any `train_v*.py` script on [Modal](https://modal.com/). Create a Modal account and authenticate locally before using it.

```bash
uv sync --group cloud
uv run modal run --detach remote.py --cmd "python -u train_v9.py --dataset ml-20m"
```

`remote.py` automatically preprocesses the requested dataset, mounts the local training scripts and `sasrec/` package, adds `--device cuda`, and persists downloaded data in a _Modal Volume_.

If you plan to fan out several ML-20M jobs with `sweep.sh` or manually, run one Modal job first and let it finish preprocessing. That warms the shared data volume and avoids multiple first-time jobs racing to download and extract the same dataset. If there are issues with dataset processing you may need to wipe the _Modal Volume_.

v4+ scripts save a checkpoint every 5 epochs (`--checkpoint-every`, 0 to disable) to `checkpoints/<script>-<dataset>-seed<N>-<hash>/`. The hash covers every argument that changes the weights. Re-running the same command resumes from the latest checkpoint, restoring the model, the optimizer and all random number generators, so the result matches an uninterrupted run. That match is exact on CPU (`tests/test_checkpoint.py`). On GPU, v7 and v8 drift in the last digits from run to run even without resuming, because the backward pass of their relative bias lookup uses CUDA atomic adds, whose order varies. On Modal, checkpoints live on their own Volume, so a preempted job picks up where it left off when Modal restarts it. `--epochs` is not part of the hash. That means a finished run can be extended by re-running it with a larger `--epochs`, and because the learning rate is constant, the result is the same as a single longer run. The hash covers arguments, not code. After changing a training script, delete its run directory (or pass a new `--checkpoint-dir`) so the run starts fresh.

You can also use a regular GPU VM from providers like [Lambda](https://lambda.ai/instances), [Lightning AI](https://lightning.ai/), or any cloud where `uv sync` can run.

This project is not sponsored.

## Roadmap

- **In-batch negatives**: use positive items from other users in the batch as harder negatives. Likely the missing piece for v10's temperature + L2 norm to work.
- **Production optimization**: KV-cache for incremental inference, FAISS for approximate nearest neighbor retrieval, mixed precision training.
- **Stochastic Length training**: randomly sample contiguous subsequences during training instead of always using the most recent items. See [PAPERS.md](PAPERS.md).
