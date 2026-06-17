# nanoSASRec

The simplest, most readable implementation of Self-Attentive Sequential Recommendation (SASRec), following the research line that made it the standard baseline and the improvements that eventually led to the Hierarchical Sequential Transduction Unit (HSTU).

The entire codebase is ~2,000 lines of Python across 10 training scripts and 3 shared modules. No file is longer than 300 lines.

Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT): every file is self-contained, every design decision cites a paper, and every step has a concrete reproduction target.

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

Train v1 on MovieLens 1M in ~7 minutes on a MacBook. Reproduce Meta's HSTU MovieLens 20M benchmark in ~4 hours on a single A100.

## What to expect

See [CURRICULUM.md](CURRICULUM.md) for a walkthrough of all 10 versions with training outputs, benchmark numbers, and detailed analysis of what works, what doesn't, and why.

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

| Step | What changes | Dataset | Key finding |
|------|-------------|---------|-------------|
| `train_v1.py` | Vanilla SASRec + BCE loss | ML-1M | BCE overconfidence problem |
| `train_v2.py` | Sampled softmax loss | ML-1M | Competition between candidates fixes ranking |
| `train_v3.py` | gBCE loss + calibration | ML-1M | Calibrated probabilities, but SSM ranks better |
| `train_v4.py` | Init, reversed positions, dropout | ML-1M | Implementation details compound to +14% NDCG |
| `train_v5.sh` | Depth sweep (2/4/8/16 blocks) | ML-20M | Depth pays off, diminishing returns after 8 |
| `train_v6.py` | ReZero residual scaling | ML-20M | 8 blocks + ReZero > 16 blocks without |
| `train_v7.py` | Relative position + temporal bias | ML-20M | Didn't help at this scale (honest negative) |
| `train_v8.py` | SiLU + gating (HSTU) | ML-20M | Architecture swap is lateral without training changes |
| `train_v9.py` | φ1 SwishLayerNorm pre-activation | ML-20M | Matches Meta's HSTU, training > architecture confirmed |
| `train_v10.py` | Temperature, L2 norm, collision masking | ML-20M | Production techniques need in-batch negatives to help |
| `train_v11.py` | Dense All Action loss (appendix) | ML-20M | Optimizes different objective than next-item eval |

Each file is runnable independently. Read them in order for the full progression, or jump to any step.

## Datasets

| Dataset | Role | Users | Items | Avg len | When used |
|---------|------|-------|-------|---------|-----------|
| ML-1M | Curriculum baseline | 6,040 | 3,416 | 163.5 | v1-v4 |
| ML-20M | Benchmark target | ~138K | ~18K | ~144 | v5-v10 |

## Benchmarks

Below is the full-catalog evaluation on ML-20M (dim=256, 8 heads, 50 epochs) from v5-v10. Every model ranks the ground truth against all 18,345 items, not a sampled subset.

| Version | What changed | HR@10 | NDCG@10 |
|---------|-------------|-------|---------|
| v5 (2 blocks) | SASRec baseline | 0.3023 | 0.1778 |
| v5 (4 blocks) | + depth | 0.3146 | 0.1865 |
| v5 (8 blocks) | + more depth | 0.3264 | 0.1966 |
| v5 (16 blocks) | + even more depth | 0.3337 | 0.2012 |
| v6 (8 blocks) | + ReZero | 0.3376 | 0.2033 |
| v7 (8 blocks) | + relative bias | 0.3208 | 0.1927 |
| v8 (8 blocks) | HSTU architecture | 0.3204 | 0.1918 |
| v9 (8 blocks) | + φ1 pre-activation | 0.3272 | 0.1967 |
| v10 (8 blocks) | + temp + L2 + collision | 0.3200 | 0.1911 |

Compared to Meta's published results (Zhai et al., ICML 2024):

| Method | HR@10 | NDCG@10 |
|--------|-------|---------|
| Meta's SASRec | 0.2889 | 0.1621 |
| Meta's HSTU | 0.3273 | 0.1895 |
| Meta's HSTU-large | 0.3556 | 0.2098 |

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

## Roadmap

- **In-batch negatives**: use positive items from other users in the batch as harder negatives. Likely the missing piece for v10's temperature + L2 norm to work.
- **Production optimization**: KV-cache for incremental inference, FAISS for approximate nearest neighbor retrieval, mixed precision training.
- **Beauty dataset**: second benchmark to validate results generalize beyond MovieLens.
- **Stochastic Length training**: randomly sample contiguous subsequences during training instead of always using the most recent items. See [PAPERS.md](PAPERS.md).