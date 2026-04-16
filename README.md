# nanoSASRec

The simplest, most readable implementation of sequential recommendation models, from SASRec to HSTU.

Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT): every file is self-contained, every design decision cites a paper, and every step has a concrete reproduction target.

## Why this exists

Most recommender systems — collaborative filtering, two-tower models, deep learning rankers — learn what a user *likes*. They struggle with what a user *needs right now*. On Amazon, someone who just bought a DSLR camera doesn't need more camera recommendations — they need a lens, then a memory card, then a camera bag. On MovieLens, a user who just watched all three Lord of the Rings films back-to-back is on a fantasy binge, not looking for the romantic comedies their five-year profile suggests. The sequence encodes intent progression that a static user profile can't capture.

This matters because real user behavior is *streaky* — people shop in projects, watch in moods, and listen in phases. Sequential models pick up on these short-term patterns while still remembering long-term preferences, which is why they dominate production recommendation at companies like Meta, YouTube, Kuaishou, and Meituan across e-commerce, video, food delivery, and live content.

Sequential recommendation models the *order* of a user's actions. RNNs (GRU4Rec, 2016) did it first, but Transformers turned out to be better at it. **SASRec** (Kang & McAuley, 2018) was the first to apply a causal Transformer decoder to item sequences, and it became the dominant baseline for the next five years.

The problem: SASRec topped out at 2 attention layers and couldn't scale deeper. Its loss function produced overconfident predictions. And for years, everyone thought a rival architecture (BERT4Rec) was better — until two 2023 papers independently proved the issue was the loss function, not the architecture.

**HSTU** (Meta, 2024) fixed the scaling problem by replacing softmax attention with SiLU, adding relative attention bias, and introducing pointwise gating — enabling power-law scaling to trillion parameters. It now powers recommendations at Meta, Kuaishou, Meituan, and others. Google/YouTube took a different path with Semantic IDs (PLUM), but the destination is the same: Transformers are the backbone of modern production recommendation. This is the GPT moment for the field.

This repo builds HSTU from scratch, one component at a time, starting from vanilla SASRec. Each step fixes one problem, shows the metric improvement, and cites the paper that figured it out. If you know LLMs: SASRec is the original Transformer, HSTU is GPT-2 — same family, but with the refinements that unlocked scaling. By v8 you'll understand causal self-attention, loss function design for large catalogs, scaling laws for recommendation, and the specific architectural decisions that separate a research baseline from a production system.

## Steps

```
train_v1.py  — vanilla SASRec + BCE                          (ML-1M)
train_v2.py  — switch to sampled softmax                     (ML-1M, Beauty)
train_v3.py  — add gBCE loss + calibration plot              (ML-1M, Beauty)
train_v4.py  — implementation refinements, decompose modules (ML-1M)
train_v5.py  — scale to ML-20M, watch depth collapse         (ML-20M)
train_v6.py  — fix residual pattern                          (ML-20M)
train_v7.py  — add relative attention bias                   (ML-20M)
train_v8.py  — SiLU + pointwise gating → HSTU               (ML-20M)
```

Each file is runnable independently. Read them in order for the full progression, or jump to any step.

## Quick start

```bash
uv sync
uv run python data.py --dataset ml-1m
uv run python train_v1.py
```

Expected output for `train_v1.py` on ML-1M:
```
  epoch   loss     P(pos)   val NDCG    val HR   time
  -------------------------------------------------------
      1   0.6102   0.5812     0.3102    0.5015   12.3s
     20   0.3341   0.9014     0.5321    0.7612   11.8s
    100   0.1284   0.9983     0.5831    0.8123   11.7s  <-- P(pos) → 0.999
    200   0.0891   0.9996     0.5897    0.8190   11.6s

  Test NDCG@10: 0.5897
  Test HR@10:   0.8190
  Target (pmixer): NDCG@10 ≈ 0.59, HR@10 ≈ 0.82
```

Watch the `P(pos)` column. It climbs to 0.9996 — the model thinks *everything* is 99.96% likely.
That's the overconfidence problem. We fix it in `train_v2.py`.

## Datasets

| Dataset | Role | Users | Items | Avg len | When used |
|---------|------|-------|-------|---------|-----------|
| ML-1M | Unit test | 6,040 | 3,416 | 163.5 | v1–v4 |
| Beauty | Sparsity stress test | 22,363 | 12,101 | 8.8 | v2–v3 |
| ML-20M | Scaling test | ~138K | ~27K | ~144 | v5–v8 |

## Papers

See [PAPERS.md](PAPERS.md) for the full research lineage — from GRU4Rec through HSTU — and which paper justifies which code decision.

## Requirements
 
Python 3.12+ and PyTorch 2.7+. That's it — no frameworks, no config systems, no custom CUDA kernels.
 
```toml
dependencies = [
    "torch>=2.7",
    "numpy>=1.24",
]
```