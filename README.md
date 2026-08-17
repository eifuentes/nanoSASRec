# nanoSASRec

The simplest, most readable implementation of Self-Attentive Sequential Recommendation (SASRec), following the research line that made it the standard baseline and the improvements that eventually led to the Hierarchical Sequential Transduction Unit (HSTU).

The official curriculum is 10 training scripts backed by a small shared `sasrec/` package.

Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT): every training version is runnable directly, every design decision cites a paper, and every version has a concrete reproduction target.

## Why this exists

Most recommender systems (collaborative filtering, two-tower models, deep learning recommender models) learn what a user *likes*. They struggle with what a user *needs right now*. On Amazon, someone who just bought a DSLR camera doesn't need more camera recommendations. They need a lens, then a memory card, then a camera bag. With movies, a user who just watched all three Lord of the Rings films back-to-back is on a fantasy binge, not looking for the romantic comedies their five-year profile suggests. The sequence encodes intent progression that a static user profile can't capture.

This matters because real user behavior is *streaky*. People shop in projects, watch in moods, and listen in phases. Sequential models pick up on these short-term patterns while still remembering long-term preferences, which is why they recently dominate production recommendation at companies like Meta, YouTube, Kuaishou, and Meituan.

Sequential recommendation models the *order* of a user's actions. RNNs (GRU4Rec, 2016) did it first, but Transformers turned out to be better at it. **SASRec** (Kang & McAuley, 2018) was the first to apply a causal Transformer decoder to item sequences.

This project builds up SASRec from its original framing all the way to its HSTU variant, one component at a time. Each step fixes one problem, shows the metric improvement, and cites the paper that figured it out. To use LLMs as an analogy: if SASRec is the original Transformer, HSTU is GPT-2. Same model lineage, but with the refinements that unlocked scaling laws. By v10 you'll understand causal self-attention, loss function design for large catalogs, scaling laws for recommendation, and the specific architectural decisions that separate a research baseline from a production system.

## Datasets

The project supports four prepared dataset outputs. MovieLens supplies the fixed datasets used by the official curriculum and benchmark tables. KuaiRand supplies recipe-driven engagement data for preprocessing validation and longer-history experiments.

| Dataset | Role | Users | Items | Avg len | Output |
|---------|------|-------|-------|---------|--------|
| MovieLens 1M | Curriculum baseline | 6,040 | 3,416 | 165.5 | `ml-1m` |
| MovieLens 20M | Published-scale benchmark target | 138,493 | 18,345 | 144.3 | `ml-20m` |
| KuaiRand-Pure | Standard-impression validation slice | 25,634 | 7,550 | 55.9 | `kuairand-pure-item1-user5` |
| KuaiRand-27K | Long-view engagement validation slice | 26,766 | 7,556 | 271.9 | recipe-stamped |

### MovieLens 1M

MovieLens 1M is the small, familiar starting point for the curriculum. It is used by v1-v4 for teaching the loss-function and implementation refinements without requiring a large machine. `data.py` downloads the archive and writes the standard split contract directly:

```bash
uv run python data.py --dataset ml-1m
uv run python train_v1.py --dataset ml-1m --epochs 200 --eval-every 20
```

The output contains human-readable movie titles, so prediction examples can display movie names instead of numeric item IDs.

### MovieLens 20M

MovieLens 20M is the official scale-up and benchmark dataset for v5-v10. Its larger user and item populations make depth, relative bias, HSTU, and training-alignment experiments meaningful and allow comparison with published full-catalog results.

```bash
uv run python data.py --dataset ml-20m
uv run python train_v9.py --dataset ml-20m --epochs 50 --eval-every 10 --device cuda
```

The `train_v5.sh` convenience wrapper prepares ML-20M and runs the 2/4/8/16-block depth sweep used by the curriculum.

### KuaiRand-Pure

KuaiRand-Pure is a compact engagement-data validation slice. It keeps standard impressions, preserves repeated interactions, and applies `item_floor=1` and `user_floor=5`. Unlike MovieLens, its item identities do not generally have human-readable titles in this repository.

```bash
uv run python data.py \
  --dataset kuairand-pure \
  --kuairand-item-floor 1 \
  --kuairand-user-floor 5
uv run python train_v9.py \
  --dataset kuairand-pure-item1-user5 \
  --epochs 50 --eval-every 10 --device cuda
```

### KuaiRand-27K

KuaiRand-27K is the long-history validation dataset. The maintained recipe keeps `long_view=1`, excludes `is_hate=1` and random impressions, uses the empirical `tab=1` condition, and applies `item_floor=555` and `user_floor=5`. Typed Parquet and DuckDB caches avoid repeatedly parsing the large raw logs during recipe experiments.

```bash
uv run python data.py \
  --dataset kuairand-27k \
  --kuairand-interaction-rule long-view-not-hate-tab1 \
  --kuairand-exclude-rand \
  --kuairand-item-floor 555 \
  --kuairand-user-floor 5

uv run python train_v9.py \
  --dataset kuairand-27k-rule-long-view-not-hate-tab1-item555-user5-exclude-rand \
  --epochs 200 --eval-every 20 --device cuda
```

KuaiRand output names include their recipe parameters and must be passed unchanged to training. Every output writes `recipe.json` next to `split.json`; analysis-only runs also write `analysis.json` and an item-frequency profile CSV. Add `--kuairand-analysis-only` to inspect a recipe before exporting training files.

## Quick start

```bash
uv sync
uv run python data.py --dataset ml-1m
uv run python train_v1.py --epochs 200 --eval-every 20  # --device cuda
```

The default quick start uses MovieLens 1M because it is small and is the first dataset in the official curriculum. Train v1 in roughly 7 minutes on a MacBook, then use the [Versions](#versions) and [Datasets](#datasets) sections to choose another dataset. Follow [CURRICULUM.md](CURRICULUM.md) for the v1-v10 commands and expected results. Reproduce the published MovieLens 20M comparisons in roughly 4 hours on a single A100.

## What to expect

See [CURRICULUM.md](CURRICULUM.md) for a walkthrough of the 10 official versions with training outputs, benchmark numbers, and detailed analysis of what works, what doesn't, and why.

## What it looks like

```
Recent history:  item_17, item_42, item_91
Ground truth:    item_108
Model top-3:
  1. item_63
  2. item_12
  3. item_77
```

The training scripts consume the same preprocessed contract for every dataset: user sequences, train/validation/test splits, item IDs, and optional item titles. MovieLens supplies human-readable movie titles; KuaiRand outputs use the available item identity metadata.

## Versions

| Version | What changes | Primary benchmark dataset | Key finding |
|------|-------------|---------|-------------|
| `train_v1.py` | Vanilla SASRec + BCE loss | ML-1M | BCE overconfidence problem |
| `train_v2.py` | Sampled softmax loss | ML-1M | Competition between candidates fixes ranking |
| `train_v3.py` | gBCE loss + calibration | ML-1M | Calibrated probabilities, but SSM ranks better |
| `train_v4.py` | Init, reversed positions, dropout | ML-1M | Implementation details compound to +14% NDCG |
| `train_v5.sh` | Depth sweep (2/4/8/16 blocks) | ML-20M | Depth pays off, diminishing returns after 8 |
| `train_v6.py` | ReZero residual scaling | ML-20M | 8 blocks + ReZero > 16 blocks without |
| `train_v7.py` | Relative position + temporal bias | ML-20M | Best model in repo, biggest single improvement |
| `train_v8.py` | SiLU + gating (HSTU) | ML-20M | Architecture swap is not a clear win at this scale |
| `train_v9.py` | φ1 SwishLayerNorm pre-activation | ML-20M | Beats the Zhai et al. (2024) HSTU result by +4.6% |
| `train_v10.py` | Temperature, L2 norm, collision masking | ML-20M | Production techniques need in-batch negatives to help |

Each Python training version is runnable independently with `--dataset <prepared-dataset>`. The table shows the dataset used for the reported result, not a hard-coded dataset restriction. `train_v5.sh` remains a convenience wrapper for the fixed ML-20M depth sweep. Read v1-v10 in order for the full progression, or jump to any version.

`train_v11.py` is an appendix experiment for Dense All Action loss. It is useful context, but not part of the official curriculum path.

## Benchmarks

The benchmark tables below only cover datasets and evaluation protocols with relevant published comparisons. The repository does not claim a published KuaiRand benchmark; KuaiRand is currently used for preprocessing validation and additional experiments.

### MovieLens 20M

These are the repository's full-catalog results on MovieLens 20M (dim=256, 8 heads, 50 epochs) from v5-v10. Every model ranks the ground truth against all 18,345 items, not a sampled subset.

| Version | What changed | HR@10 | NDCG@10 |
|---------|-------------|-------|---------|
| v5 (2 blocks) | SASRec baseline | 0.3009 | 0.1760 |
| v5 (4 blocks) | + depth | 0.3149 | 0.1869 |
| v5 (8 blocks) | + more depth | 0.3237 | 0.1941 |
| v5 (16 blocks) | + even more depth | 0.3350 | 0.2026 |
| v6 (8 blocks) | + ReZero | 0.3365 | 0.2020 |
| **v7 (8 blocks)** | **+ relative bias** | **0.3457** | **0.2085** |
| v8 (8 blocks) | HSTU architecture | 0.3339 | 0.2006 |
| v9 (8 blocks) | + φ1 pre-activation | 0.3425 | 0.2071 |
| v10 (8 blocks) | + temp + L2 + collision | 0.3355 | 0.2016 |

### Published MovieLens comparisons

For context, the following MovieLens 20M full-catalog results were published by Zhai et al. (2024):

| Method | HR@10 | NDCG@10 |
|--------|-------|---------|
| SASRec | 0.2889 | 0.1621 |
| HSTU | 0.3273 | 0.1895 |
| HSTU-large | 0.3556 | 0.2098 |

See [CURRICULUM.md](CURRICULUM.md) for the full analysis from v1 to v10.

## Papers

See [PAPERS.md](PAPERS.md) for the full research lineage from GRU4Rec through HSTU and which paper justifies which code decision.

## Requirements

Python 3.12+, PyTorch 2.7+, and DuckDB for KuaiRand preprocessing. No frameworks, no config systems, no custom CUDA kernels.

```toml
dependencies = [
    "torch>=2.7",
    "numpy>=1.24",
    "duckdb>=1.1",
]
```

If you plan to edit the repo, install the dev tools and run Ruff before committing changes:

```bash
uv sync --group dev
uv run ruff format .
uv run ruff check .
```

## Remote GPU Training

For longer runs on ML-20M or KuaiRand, `remote.py` can launch CPU preprocessing jobs and GPU training jobs on [Modal](https://modal.com/). Create a Modal account and authenticate locally before using it. Every remote command must include `--dataset`; the wrapper maps the dataset to its `/volumes/<dataset>` path and overrides any supplied `--data-dir`.

```bash
uv sync --group cloud
uv run modal run --detach remote.py::prep_job --cmd "python -u data.py --dataset kuairand-27k --kuairand-interaction-rule long-view-not-hate-tab1 --kuairand-exclude-rand --kuairand-item-floor 555 --kuairand-user-floor 5"
uv run modal run --detach remote.py::train --cmd "python -u train_v9.py --dataset kuairand-27k-rule-long-view-not-hate-tab1-item555-user5-exclude-rand --epochs 200 --eval-every 20 --device cuda"
```

Preprocessing and training are intentionally separate. `remote.py::prep_job` runs CPU/IO-heavy `data.py` work, persists downloaded data plus DuckDB/Parquet caches in a per-dataset _Modal Volume_, and can be detached safely from your laptop. `remote.py::train` assumes the dataset is already prepared, mounts the same volume, and runs the training command on a GPU. Recipe-stamped KuaiRand datasets reuse the base dataset volume, so a training command points at the stamped output while the volume remains `kuairand-pure` or `kuairand-27k`.

For the confirmed Pure slice:

```bash
uv run modal run --detach remote.py::prep_job --cmd "python -u data.py --dataset kuairand-pure --kuairand-item-floor 1 --kuairand-user-floor 5"
```

If you plan to fan out several jobs with `sweep.sh` or manually, run one Modal prep job first and let it finish preprocessing. The checked-in `sweep.sh` prepares and launches the official ML-1M/ML-20M curriculum; prepare KuaiRand explicitly using the commands above before launching KuaiRand training jobs. This warms the dataset-specific volume and avoids multiple first-time jobs racing to download, extract, or convert the same dataset. If there are issues with dataset processing you may need to wipe the relevant _Modal Volume_.

You can also use a regular GPU VM from providers like [Lambda](https://lambda.ai/instances), [Lightning AI](https://lightning.ai/), or any cloud where `uv sync` can run.

This project is not sponsored.

## Roadmap

- **In-batch negatives**: use positive items from other users in the batch as harder negatives. Likely the missing piece for v10's temperature + L2 norm to work.
- **Production optimization**: KV-cache for incremental inference, FAISS for approximate nearest neighbor retrieval, mixed precision training.
- **Stochastic Length training**: randomly sample contiguous subsequences during training instead of always using the most recent items. See [PAPERS.md](PAPERS.md).
