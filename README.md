# nanoSASRec

The simplest, most readable implementation of Self-Attentive Sequential Recommendation (SASRec), following the research line that made it the standard baseline — and the improvements that eventually led to the Hierarchical Sequential Transduction Unit (HSTU).

Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT): every file is self-contained, every design decision cites a paper, and every step has a concrete reproduction target.

## Why this exists

Most recommender systems — collaborative filtering, two-tower models, deep learning recommender models (DLRM) — learn what a user *likes*. They struggle with what a user *needs right now*. On Amazon, someone who just bought a DSLR camera doesn't need more camera recommendations — they need a lens, then a memory card, then a camera bag. With movies, a user who just watched all three Lord of the Rings films back-to-back is on a fantasy binge, not looking for the romantic comedies their five-year profile suggests. The sequence encodes intent progression that a static user profile can't capture.

This matters because real user behavior is *streaky* — people shop in projects, watch in moods, and listen in phases. Sequential models pick up on these short-term patterns while still remembering long-term preferences, which is why they dominate production recommendation at companies like Meta, YouTube, Kuaishou, and Meituan.

Sequential recommendation models the *order* of a user's actions. RNNs (GRU4Rec, 2016) did it first, but Transformers turned out to be better at it. **SASRec** (Kang & McAuley, 2018) was the first to apply a causal Transformer decoder to item sequences.

This project builds up SASRec from its original framing all the way to its HSTU variant, one component at a time. Each step fixes one problem, shows the metric improvement, and cites the paper that figured it out. To use LLMs as an analogy: if SASRec is the original Transformer, HSTU is GPT-2 — same model lineage, but with the refinements that unlocked scaling. By v8 you'll understand causal self-attention, loss function design for large catalogs, scaling laws for recommendation, and the specific architectural decisions that separate a research baseline from a production system.

## Steps

| Step | Change | Dataset |
|------|--------|---------|
| `train_v1.py` | vanilla SASRec + BCE | ML-1M |
| `train_v2.py` | switch to sampled softmax | ML-1M, Beauty |
| `train_v3.py` | add gBCE loss + calibration plot | ML-1M, Beauty |
| `train_v4.py` | implementation refinements, decompose modules | ML-1M |
| `train_v5.py` | scale to ML-20M, watch depth collapse | ML-20M |
| `train_v6.py` | fix residual pattern | ML-20M |
| `train_v7.py` | add relative attention bias | ML-20M |
| `train_v8.py` | SiLU + pointwise gating → HSTU | ML-20M |

Each file is runnable independently. Read them in order for the full progression, or jump to any step.

## Quick start

```bash
uv sync
uv run python data.py --dataset ml-1m
uv run python train_v1.py --epochs 200 --eval-every 20  # --device cuda
```

## What to expect

Check out [CURRICULUM.md](CURRICULUM.md) to walk through all 8 versions of this project that build up from each other and what to expect.

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