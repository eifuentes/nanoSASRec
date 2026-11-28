# The SASRec Research Lineage

Every design decision in this repo traces back to a specific paper. This document maps code to research, plus the earlier papers those decisions build on, so you can trace the lineage.

The story: SASRec replaces RNNs with Transformers, BERT4Rec challenges it and the
community splits, the loss function is identified as the real problem, papers argue SASRec
fails to scale with depth (in this repo, that traces back to the block wiring in its released
code), PinnerFormer bridges to production, HSTU carries the line into trillion-parameter scale, and
eSASRec shows that SASRec with the right loss and layers is competitive with HSTU.

## Foundation (2016-2018)

- **Hidasi et al., ICLR 2016** (1511.06939): GRU4Rec. First RNN for session-based recommendation. The baseline SASRec was built to replace.
- **Vaswani et al., NeurIPS 2017** (1706.03762): "Attention Is All You Need." The Transformer architecture. SASRec adapts the decoder-only (causal) variant.
- **Kang & McAuley, ICDM 2018** (1808.09781): **SASRec.** Causal self-attention over item sequences for next-item prediction. Trained with BCE loss and 1 negative sample. The architecture this entire repo is built around. The paper's block formula is pre-LN, `x + Dropout(g(LayerNorm(x)))`, but the authors' released code is wired differently, and that wiring collapses at 8+ blocks on ML-20M (CURRICULUM v4 and v5).
- **Sun et al., CIKM 2019** (1904.06690): **BERT4Rec.** Bidirectional attention with masked-item prediction (Cloze objective). Claimed superiority over SASRec, setting up the debate resolved below.

## The SASRec vs. BERT4Rec debate (2019-2022)

- **Petrov & Macdonald, RecSys 2022** (2207.07483): BERT4Rec replicability study. Reviewed 370 citing papers, found results inconsistent. BERT4Rec wins only 64% of comparisons. Conclusion: BERT4Rec is SOTA *if trained long enough*, but most papers undertrained it. Its ML-1M preprocessing (5-core filtering, leave-one-out) is the one this repo uses for ML-1M.

## The loss function is the real problem (2023-2025)

Two independent groups reached the same conclusion in the same year: SASRec's
underperformance is caused by its loss function, not its architecture.

- **Jean et al., ACL-IJCNLP 2015** (1412.2007): Original Sampled Softmax for NLP (large vocabulary neural machine translation). The technique the "Dross" paper adapts for recommendation.
- **Klenitskiy & Vasilev, RecSys 2023** (2309.07602): "Turning Dross Into Gold Loss." Shows that SASRec trained with cross-entropy loss (full or sampled softmax) matches or exceeds BERT4Rec while training faster. The paper that showed architecture wasn't the bottleneck.
- **Petrov & Macdonald, RecSys 2023** (2308.07192): **gSASRec.** Defines overconfidence formally: BCE + negative sampling causes predicted probabilities to approach 1.0. Proposes gBCE loss, which raises the positive's predicted probability to a power β < 1 set by a calibration parameter t and the negative sampling rate. Our v3 implements it as in the paper's reference code (github.com/asash/gSASRec-pytorch). **Best Full Paper Award, RecSys 2023.** The journal version (ACM TORS 2025, doi:10.1145/3699521) adds full proofs and recommends t ∈ [0.75, 0.9] with 128-256 negatives.
- **Zhai et al., KDD 2023** (2306.04039): "Revisiting Neural Retrieval on Accelerators." Applies sampled softmax to large-scale retrieval. Same first author as HSTU. The Zhai et al. (2024) SASRec benchmark uses this loss.

## Implementation details that move metrics 2-5%

- **Huang, arXiv 2025** (2504.09596): "Revisiting Self-Attentive Sequential Recommendation." Positional embedding alignment (steps-back-from-prediction), padding embedding must be frozen at zero, advocacy for native PyTorch attention primitives.
- **D'Ercoli et al., arXiv 2025** (2506.14692): Replicability study of BSARec vs SASRec on a common codebase (EasyRec). Documents that normalization order, dropout placement, and initialization cause 2-5% metric swings.
- **Xiong et al., ICML 2020** (2002.04745): "On Layer Normalization in the Transformer Architecture." The general Pre-LN vs Post-LN analysis.
- **pmixer/SASRec.pytorch, May 2025** ([#47](https://github.com/pmixer/SASRec.pytorch/issues/47), [results](https://github.com/Tiny-Snow/SASRec.pytorch/blob/main/Result_Norm.md)): Points out that SASRec's released code doesn't match the paper's pre-LN formula, and compares the original wiring with standard pre-LN and post-LN at 2 and 3 blocks, where the differences are small.

## Depth scaling and residual stabilization (2020-2023)

- **Peng et al., 2022 and TKDD 2023** (2209.07997, 2311.07742): argue that deeper SASRec doesn't improve. Attention spreads out toward uniform in early blocks, and item representations grow more alike with depth (over-smoothing), so SASRec's best depth is stuck at about 2 blocks. Our pre-LN v5 sweep keeps improving to 16 blocks.
- **Bachlechner et al., arXiv 2020** (2003.04887): **ReZero.** Learnable residual scaling initialized to zero. Each block starts as identity and gradually turns on. Our v6 uses this to make 8 blocks match 16 blocks without it.
- **De & Smith, NeurIPS 2020** (2002.10444): SkipInit. The same idea for ResNets: a scalar on each residual branch that starts at zero.
- **Touvron et al., ICCV 2021** (2103.17239): CaiT's LayerScale. One learned scale per channel on each residual branch, starting small rather than at zero.
- **LinkedIn, arXiv 2025** (2502.03417): "From Features to Transformers." Its LiGR layer gates the residual branch with `σ(W·x)`, computed from the input. eSASRec uses it; our v6 accepts `--residual-gate ligr` but doesn't implement it yet ([#42](https://github.com/eifuentes/nanoSASRec/issues/42)).

## Evaluation methodology

- **Krichene & Rendle, KDD 2020**: "On Sampled Metrics for Item Recommendation." Sampled metrics (100 negatives) are biased and can reverse model rankings. Use full-catalog evaluation when feasible.
- **Rendle, Zhang & Koren, 2019** (1905.01395): "On the Difficulty of Evaluating Baselines." Carefully tuned vanilla MF beats most newly proposed methods. The reason this repo insists on getting implementation details right before claiming improvements.
- **Tikhonovich et al., RecSys 2025** (2508.06450): eSASRec's shared benchmark split, which our ML-20M protocol follows. See [eSASRec](#esasrec-2025) below.

## The bridge to HSTU (2022-2024)

- **Li, Wang & McAuley, WSDM 2020** (doi: 10.1145/3336191.3371786): **TiSASRec.** Time Interval Aware Self-Attention for Sequential Recommendation. Models absolute positions and the time intervals between interactions, not just item order. Our v7's temporal bias buckets draw from this idea.
- **Pancha et al., KDD 2022** (2205.04507): **PinnerFormer.** Pinterest's industrial SASRec-style model. Introduces "Dense All Action" loss (predict all future actions, not just the next item), which our appendix `train_v11.py` tries. Pre-LN residuals, sampled softmax.
- **Guo, Wang et al., arXiv 2024** (2412.00714): "Scaling New Frontiers." Compares HSTU, Llama, GPT, and SASRec by varying depth, and reports SASRec collapsing from 4 blocks on ML-20M. Llama-style residuals and relative attention bias restore scaling. Their SASRec applies the residual "after layer normalization", like the authors' code; CURRICULUM v5 compares their results with ours.
- **Zhai et al., ICML 2024** (2402.17152): **HSTU / Generative Recommenders.** "Actions Speak Louder than Words." Replaces softmax attention with SiLU, adds relative attention bias (position + timestamp), adds U(X) gating. Trillion-parameter scale with power-law scaling. Also introduces M-FALCON, an inference algorithm that scores many candidate items in one pass by computing the user's history once. The north star. Meta's [generative-recommenders](https://github.com/meta-recsys/generative-recommenders) is the official code and the reference for our HSTU versions. Its ML-20M numbers come from a split that orders tied timestamps differently from ours (see [eSASRec](#esasrec-2025)).
- **FuXi-α, WWW 2025** (2502.03036): Extends the HSTU block with a Modified Feed-Forward Network (MFFN) using SwiGLU in a two-stage architecture. One of the re-run rows in the README's published results.

## eSASRec (2025)

- **Tikhonovich et al., RecSys 2025** (2508.06450): **eSASRec.** "Enhancing Transformer-based Recommendations in a Modular Fashion." Asks whether the separate improvements made to SASRec add up, and finds one strong combination: SASRec's shifted-sequence training objective, sampled softmax loss, and LiGR Transformer layers (pre-LN, with a learned sigmoid gate on each residual branch and a SwiGLU feed-forward). That is close to the path this repo takes from v2 to v6, with LiGR's gate where v6 uses ReZero ([#42](https://github.com/eifuentes/nanoSASRec/issues/42)). The changes are simple and need no extra features, so the authors propose it as an easy, strong baseline. In a preliminary study on academic benchmarks it beats ActionPiece by 23% NDCG@10. In their production-like evaluation it sits on the accuracy-coverage Pareto frontier with HSTU, though there it doesn't beat HSTU.
- **Its benchmark.** The paper re-runs HSTU and FuXi-α with their original code on one shared split, and finds that HSTU's preprocessing reorders tied timestamps, changing about 25% of the ML-20M test set. Our ML-20M split matches theirs, and the README's published-results table takes these numbers and eSASRec's own from the [transformer_benchmark](https://github.com/blondered/transformer_benchmark) README. Its 8-layer SASRec baseline runs on a port of the SASRec authors' code and collapses (CURRICULUM v5).

## Not implemented yet

- **Stochastic Length Training** (Zhai et al., ICML 2024): During training, randomly sample a contiguous subsequence of each user's history instead of always using the most recent items. At Meta's scale, with sequences of 8,192+ items, this makes long-sequence training tractable. On ML-20M, where sequences average 144 items, it may not move metrics ([#28](https://github.com/eifuentes/nanoSASRec/issues/28)).
