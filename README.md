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

The default quick start uses MovieLens 1M because it is small and is the first dataset in the official curriculum. Train v1 in roughly 7 minutes on a MacBook, then use the [Versions](#versions) and [Datasets](#datasets) tables to choose another dataset. Follow [CURRICULUM.md](CURRICULUM.md) for the v1-v10 commands and expected results. Reproduce the Zhai et al. (2024) HSTU MovieLens 20M benchmark in roughly 4 hours on a single A100.

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

## Datasets

| Dataset | Role | Users | Items | Avg len | Output |
|---------|------|-------|-------|---------|--------|
| ML-1M | Curriculum baseline | 6,040 | 3,416 | 165.5 | `ml-1m` |
| ML-20M | Published-scale benchmark target | 138,493 | 18,345 | 144.3 | `ml-20m` |
| KuaiRand-Pure | Confirmed FuXi-gamma/Pure validation slice | 25,634 | 7,550 | 55.9 | `kuairand-pure-item1-user5` |
| KuaiRand-27K | Long-view engagement validation slice | 26,766 | 7,556 | 271.9 | recipe-stamped |

MovieLens is downloaded and converted directly into the standard split format. KuaiRand preprocessing uses typed Parquet plus DuckDB so large 27K recipe sweeps do not repeatedly parse 322M CSV rows. The confirmed Pure recipe is all standard impressions, repeated interactions preserved, `item_floor=1`, and `user_floor=5`. The 27K long-sequence slice is `long_view=1`, `is_hate!=1`, empirical `tab=1`, `is_rand!=1`, `item_floor=555`, and `user_floor=5`. It is a principled long-view engagement slice, not a FuXi-alpha checksum reproduction.

Every KuaiRand output writes a `recipe.json` next to `split.json`. Analysis-only runs also write `analysis.json` plus an item-frequency profile CSV; those files are the provenance trail for choices like the 27K `item_floor=555` boundary. Recipe-stamped dataset names must be passed unchanged to training, for example `kuairand-27k-rule-long-view-not-hate-tab1-item555-user5-exclude-rand`.

### Prepare And Train KuaiRand

Use the Pure slice for a compact engagement dataset:

```bash
uv run python data.py --dataset kuairand-pure --kuairand-item-floor 1 --kuairand-user-floor 5
uv run python train_v9.py --dataset kuairand-pure-item1-user5 --epochs 50 --eval-every 10 --device cuda
```

Use the 27K recipe for longer sequences and explicit interaction filtering:

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

To inspect a recipe before exporting training files, add `--kuairand-analysis-only`. This writes recipe diagnostics and item-frequency profiles without producing the Python training split. KuaiRand recipes are configurable experiments, so their metrics should not be compared directly with the fixed ML-1M or ML-20M benchmark numbers.

## Benchmarks

Below is the official full-catalog evaluation on ML-20M (dim=256, 8 heads, 50 epochs) from v5-v10. Every model ranks the ground truth against all 18,345 items, not a sampled subset. These are the reproducibility results for the MovieLens curriculum; KuaiRand is currently provided for preprocessing validation and additional experiments rather than a published benchmark table.

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

Compared to Zhai et al. (2024) published results:

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
