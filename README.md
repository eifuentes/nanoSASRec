# nanoSASRec

The simplest, most readable implementation of Self-Attentive Sequential Recommendation (SASRec), following the research line that made it the standard baseline and the improvements that eventually led to the Hierarchical Sequential Transduction Unit (HSTU).

The official curriculum is 10 versions (nine training scripts plus the v5 depth sweep) backed by a small shared `sasrec/` package.

Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT): every training version is runnable directly, every design decision cites a paper, and every version has a concrete reproduction target.

## Why this exists

Most recommender systems (collaborative filtering, two-tower models, deep learning recommender models) learn what a user *likes*. They struggle with what a user *needs right now*. On Amazon, someone who just bought a DSLR camera doesn't need more camera recommendations. They need a lens, then a memory card, then a camera bag. With movies, a user who just watched all three Lord of the Rings films back-to-back is on a fantasy binge, not looking for the romantic comedies their five-year profile suggests. The sequence encodes intent progression that a static user profile can't capture.

This matters because real user behavior is *streaky*. People shop in projects, watch in moods, and listen in phases. Sequential models pick up on these short-term patterns while still remembering long-term preferences, which is why they have become the standard in production recommendation at companies like Meta, YouTube, Kuaishou, and Meituan.

Sequential recommendation models the *order* of a user's actions. GRU4Rec (2016) brought RNNs to it, and Transformers turned out to be better at it. **SASRec** (Kang & McAuley, 2018) was the first to apply a causal Transformer decoder to item sequences.

This project builds up SASRec from its original framing all the way to its HSTU variant, one component at a time. Each step changes one thing, measures what it does, and cites the paper behind it. Some steps don't help here, and the docs say so. To use LLMs as an analogy: if SASRec is the original Transformer, HSTU is GPT-2. Same model lineage, but with the refinements that unlocked scaling laws. By v10 you'll understand causal self-attention, loss design for large catalogs, how depth scales, and which production techniques do and don't transfer to a small benchmark.

## Quick start

```bash
uv sync
uv run python data.py --dataset ml-1m
uv run python train_v1.py --epochs 200 --eval-every 20 --device mps  # or cpu, cuda
```

Train v1 on MovieLens 1M in about 8 minutes on a MacBook GPU (`mps`). From there, use the [Versions](#versions) and [Datasets](#datasets) tables to see how the curriculum progresses, then follow [CURRICULUM.md](CURRICULUM.md) for the v1-v10 commands and expected results. The 8-block ML-20M versions take about 3 to 4 hours for 50 epochs on one A10 GPU (the v5 depth sweep runs from 1 to 6 hours per depth).

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
| `train_v3.py` | gBCE loss + calibration | ML-1M | Calibration needs 2× the epochs; sampled softmax still ranks better |
| `train_v4.py` | Init, reversed positions, dropout | ML-1M | Implementation details compound to +12% NDCG (sampled eval) |
| `train_v5.sh` | Depth sweep (2/4/8/16 blocks) | ML-20M | Depth pays off, with diminishing returns |
| `train_v6.py` | ReZero residual scaling | ML-20M | 8 blocks + ReZero matches 16 blocks without |
| `train_v7.py` | Relative position + temporal bias | ML-20M | Best model in repo |
| `train_v8.py` | SiLU + gating (HSTU) | ML-20M | Architecture swap is not a clear win at this scale |
| `train_v9.py` | φ1 SwishLayerNorm pre-activation | ML-20M | Recovers most of v8's gap to v7 |
| `train_v10.py` | Temperature, L2 norm, collision masking | ML-20M | Didn't help in our setup |

Each version is runnable independently. Read v1-v10 in order for the full progression, or jump to any version.

Terms in the table, in short (CURRICULUM.md explains each one where it's introduced):
- **BCE** (binary cross-entropy): scores each item as its own yes/no question.
- **Sampled softmax**: the true item competes with a set of random items for probability.
- **gBCE** (generalized BCE, from gSASRec): BCE with the positive's term reweighted to make up for sampling only a few negatives.
- **ReZero**: each block's contribution starts at zero and is learned.
- **HSTU**: Meta's 2024 recommender block, which replaces softmax attention with SiLU attention and a gate.
- **φ1**: HSTU's nonlinearity on the projections before attention.
- **A10**: the 24 GB NVIDIA GPU used for every ML-20M run.

`train_v11.py` is an appendix experiment for Dense All Action loss. It is useful context, but not part of the official curriculum path.

## Datasets

| Dataset | Role | Users | Items | Avg len | When used |
|---------|------|-------|-------|---------|-----------|
| ML-1M | Curriculum baseline | 6,040 | 3,416 | 165.5 | v1-v4 |
| ML-20M | Benchmark target | 138,493 | 26,744 | 144.4 | v5-v10 |

ML-20M follows the HSTU (Zhai et al. 2024) and eSASRec (Tikhonovich et al. 2025) protocol. There is no k-core filtering (dropping items or users with fewer than k interactions), so the full 26,744-item catalog is kept, and interactions with tied timestamps keep their `ratings.csv` order. `data.py` verifies each download against the MD5 GroupLens publishes, checks the processed statistics against the published numbers, and writes a `metadata.json` next to `split.json` that records the source and protocol. Up to v0.2.1 (the previous release), nanoSASRec dropped items with fewer than 5 ratings (18,345 items). That made ranking easier and made the numbers not comparable with published ML-20M results.

From v4 onward, validation and test metrics rank the ground truth against the full item catalog, excluding items already in the user's history. v1-v3 keep sampled evaluation (100 negatives) for both validation and test, because they teach loss functions.

## Benchmarks

Full-catalog evaluation on ML-20M (dim=256, 8 heads, 50 epochs, one A10 GPU). Every model ranks the ground truth against all 26,744 items, not a sampled subset. v7 and v9 are the mean ± standard deviation over seeds 42, 43 and 44 (`--seed`). The other rows use seed 42. HR@10 and NDCG@10 are explained in [Reading the Metrics](CURRICULUM.md#reading-the-metrics).

| Version | What changed | HR@10 | NDCG@10 |
|---------|-------------|-------|---------|
| v5 (2 blocks) | SASRec baseline | 0.2938 | 0.1713 |
| v5 (4 blocks) | + depth | 0.3098 | 0.1830 |
| v5 (8 blocks) | + more depth | 0.3196 | 0.1912 |
| v5 (16 blocks) | + even more depth | 0.3280 | 0.1970 |
| v6 (8 blocks) | + ReZero | 0.3287 | 0.1973 |
| **v7 (8 blocks)** | **+ relative bias** | **0.3380 ± 0.0009** | **0.2025 ± 0.0011** |
| v8 (8 blocks) | HSTU architecture | 0.3236 | 0.1929 |
| v9 (8 blocks) | + φ1 pre-activation | 0.3354 ± 0.0009 | 0.2008 ± <0.0001 |
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
| Tikhonovich et al. (2025) | FuXi-α, recomputed (8 blocks) | 0.3468 | 0.2043 |

eSASRec is SASRec with sampled softmax and a gated Transformer layer (LiGR), from a paper that re-benchmarks sequential recommenders on a shared split. Its numbers come from the [transformer_benchmark](https://github.com/blondered/transformer_benchmark) README. Its HSTU and FuXi-α rows are those models' original code re-run on the same order-preserving split that `data.py` produces.

The eSASRec rows are the closest comparison because they use the same split and the same evaluation: full-catalog ranking with the user's earlier items masked. After 50 epochs, v7 is 0.8 points of HR@10 below eSASRec and the recomputed HSTU at 8 blocks. eSASRec trains for up to 100 epochs, though, and [Training longer](#training-longer) shows v7 draws level with both when given the same budget. Our HSTU versions trail the recomputed HSTU by more: 1.1 points for v9 and 2.3 for v8 at 50 epochs, and 0.4 and 1.5 at 100. Our HSTU also differs from the reference code in small ways: the reference divides the SiLU attention by sequence length, and its φ1 is SiLU on the projection of a layer-normed input rather than SwishLayerNorm. We haven't tested which of these matter. Zhai et al.'s own rows come from preprocessing that reorders tied timestamps, so about 25% of their test interactions differ from ours.

### Training longer

The 50-epoch table compares versions under one fixed budget. It is not where training stops: every version is still improving at epoch 50. Re-running five of them, plus v7 with eSASRec's settings, for 100 epochs (seed 42; v7 is the mean ± std over seeds 42-44):

| Version | HR@10, 50 epochs | HR@10, 100 epochs | NDCG@10, 100 epochs |
|---------|------------------|-------------------|---------------------|
| v5 (8 blocks) | 0.3196 | 0.3297 | 0.1976 |
| **v7 (8 blocks)** | 0.3380 ± 0.0009 | **0.3473 ± 0.0013** | **0.2098 ± 0.0005** |
| v7 with eSASRec's settings | | 0.3476 | 0.2091 |
| v8 (8 blocks) | 0.3236 | 0.3313 | 0.1976 |
| v9 (8 blocks) | 0.3344 | 0.3426 | 0.2061 |
| v10 (8 blocks) | 0.3269 | 0.3336 | 0.1992 |

v9's 50-epoch number here is seed 42 alone, so it differs slightly from the three-seed mean in the table above.

- Every version gains 0.7 to 1.0 points of HR@10 from the extra 50 epochs. The SASRec versions (v5, v7) gain about 1 point, the HSTU versions (v8 to v10) 0.7 to 0.8.
- At 100 epochs, v7 matches eSASRec (0.3457 HR@10, 0.2107 NDCG@10) and the recomputed HSTU and FuXi-α (0.3464 and 0.3468). Its HR@10 is 0.1 to 0.2 points higher, about one standard deviation across seeds, so treat it as a tie, not a win. On NDCG@10 it is 0.1 points below eSASRec and 0.6 above the recomputed HSTU.
- eSASRec's settings (4× feed-forward, 16 heads, 256 negatives) don't help v7. They add 30% more parameters and take 1.8× longer per epoch for the same accuracy.
- The order doesn't change: v7 > v9 > v10 > v8 > v5. v9, the best HSTU version, closes to 0.4 points behind the recomputed HSTU; v8 is still 1.5 behind.
- v8's lead over v5 at the same depth shrinks from 0.4 to under 0.2 points. HSTU learns faster early on, but SASRec catches up.

Each model is scored on test once, after its last epoch. For most runs the best validation HR@10 also came at the last epoch, so picking the checkpoint by validation would not change their numbers. The exceptions are v7 seed 44 and v5 (best at epoch 95) and v8 (best at epoch 90), each within 0.3 points of their last epoch.

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

You can also use a regular GPU VM from providers like [Lambda](https://lambda.ai/instances), [Lightning AI](https://lightning.ai/), or any cloud where `uv sync` can run.

This project is not sponsored.

## Roadmap

- **In-batch negatives**: use positive items from other users in the batch as harder negatives. Worth testing with v10's temperature and L2 norm, though Zhai et al. get their ML-20M results with random negatives.
- **Production optimization**: KV-cache for incremental inference, FAISS for approximate nearest neighbor retrieval, mixed precision training.
- **Stochastic Length training**: randomly sample contiguous subsequences during training instead of always using the most recent items. See [PAPERS.md](PAPERS.md).
