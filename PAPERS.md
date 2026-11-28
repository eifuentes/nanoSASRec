# The SASRec Research Lineage

Every design decision in this repo traces back to a specific paper. This document maps code to research.

The story: SASRec replaces RNNs with Transformers, BERT4Rec challenges it and the
community splits, the loss function is identified as the real problem, papers argue SASRec
fails to scale with depth (in this repo, that traces back to the block wiring in its released
code), PinnerFormer bridges to production, and HSTU carries the line into trillion-parameter scale.

## Foundation (2016-2018)

- **Hidasi et al., ICLR 2016** (1511.06939): GRU4Rec. First RNN for session-based recommendation. The baseline SASRec was built to replace.
- **Vaswani et al., NeurIPS 2017** (1706.03762): "Attention Is All You Need." The Transformer architecture. SASRec adapts the decoder-only (causal) variant.
- **Kang & McAuley, ICDM 2018** (1808.09781): **SASRec.** Causal self-attention over item sequences for next-item prediction. Trained with BCE loss and 1 negative sample. The architecture this entire repo is built around. The paper's block formula is pre-LN, `x + Dropout(g(LayerNorm(x)))`, but the authors' released code adds the residual back from `LayerNorm(x)` and feeds raw keys and values to attention. The mismatch looks like a side effect of reusing a post-LN Transformer's functions, and [pmixer/SASRec.pytorch#47](https://github.com/pmixer/SASRec.pytorch/issues/47) pointed it out in 2025. That code wiring collapses at 8+ blocks on ML-20M (CURRICULUM v4 and v5).
- **Sun et al., CIKM 2019** (1904.06690): **BERT4Rec.** Bidirectional attention with masked-item prediction (Cloze objective). Claimed superiority over SASRec, setting up the debate resolved below.

## The SASRec vs. BERT4Rec debate (2019-2022)

- **Petrov & Macdonald, RecSys 2022** (2207.07483): BERT4Rec replicability study. Reviewed 370 citing papers, found results inconsistent. BERT4Rec wins only 64% of comparisons. Conclusion: BERT4Rec is SOTA *if trained long enough*, but most papers undertrained it. Its ML-1M preprocessing (5-core filtering, leave-one-out) is the one this repo uses for ML-1M.

## The loss function is the real problem (2023-2025)

Two independent groups reached the same conclusion in the same year: SASRec's
underperformance is caused by its loss function, not its architecture.

- **Klenitskiy & Vasilev, RecSys 2023** (2309.07602): "Turning Dross Into Gold Loss." Shows that SASRec trained with cross-entropy loss (full or sampled softmax) matches or exceeds BERT4Rec while training faster. The paper that showed architecture wasn't the bottleneck.
- **Petrov & Macdonald, RecSys 2023** (2308.07192): **gSASRec.** Defines overconfidence formally: BCE + negative sampling causes predicted probabilities to approach 1.0. Proposes gBCE loss, which raises the positive's predicted probability to a power β < 1 set by a calibration parameter t and the negative sampling rate. Our v3 implements it as in the paper's reference code (github.com/asash/gSASRec-pytorch). **Best Full Paper Award, RecSys 2023.**
- **Petrov & Macdonald, ACM TORS 2025** (doi:10.1145/3699521): Expanded journal version of gSASRec. Full proofs, calibration error drops from 0.966 to 0.01. Recommends t ∈ [0.75, 0.9] with 128-256 negatives.
- **Jean et al., ACL-IJCNLP 2015** (1412.2007): Original Sampled Softmax for NLP (large vocabulary neural machine translation). The technique the "Dross" paper adapts for recommendation.
- **Zhai et al., KDD 2023** (2306.04039): "Revisiting Neural Retrieval on Accelerators." Applies sampled softmax to large-scale retrieval. Same first author as HSTU. The Zhai et al. (2024) SASRec benchmark uses this loss.

## Implementation details that move metrics 2-5%

- **Huang, arXiv 2025** (2504.09596): "Revisiting Self-Attentive Sequential Recommendation." Positional embedding alignment (steps-back-from-prediction), padding embedding must be frozen at zero, advocacy for native PyTorch attention primitives.
- **D'Ercoli et al., arXiv 2025** (2506.14692): Replicability study of BSARec vs SASRec on a common codebase (EasyRec). Documents that normalization order, dropout placement, and initialization cause 2-5% metric swings.
- **Xiong et al., ICML 2020** (2002.04745): "On Layer Normalization in the Transformer Architecture." The general Pre-LN vs Post-LN analysis.
- **pmixer/SASRec.pytorch, May 2025** ([#47](https://github.com/pmixer/SASRec.pytorch/issues/47), [results](https://github.com/Tiny-Snow/SASRec.pytorch/blob/main/Result_Norm.md)): Points out that SASRec's released code doesn't match the paper's pre-LN formula, and compares the original wiring with standard pre-LN and post-LN at 2 and 3 blocks, where the differences are small.

## Depth scaling and residual stabilization (2020-2023)

- **Bachlechner et al., arXiv 2020** (2003.04887): **ReZero.** Learnable residual scaling initialized to zero. Each block starts as identity and gradually turns on. Our v6 uses this to make 8 blocks match 16 blocks without it.
- **De & Smith, NeurIPS 2020** (2002.10444): SkipInit. The same idea for ResNets: a scalar on each residual branch that starts at zero.
- **Touvron et al., ICCV 2021** (2103.17239): CaiT's LayerScale. One learned scale per channel on each residual branch, starting small rather than at zero.
- **LinkedIn, arXiv 2025** (2502.03417): "From Features to Transformers." Its LiGR layer gates the residual branch with `σ(W·x)`, computed from the input. eSASRec uses it; our v6 accepts `--residual-gate ligr` but doesn't implement it yet ([#42](https://github.com/eifuentes/nanoSASRec/issues/42)).
- **Peng et al., 2022** (2209.07997): RAM (Recursive Attentive Methods). Identifies the "localization-deficit": attention distributions approach uniformity in early blocks, item representations diverge from originals. SASRec's optimal depth is stuck at 2 layers. (Our pre-LN v5 sweep keeps improving to 16 blocks; see Guo et al. below.)
- **Peng et al., TKDD 2023** (2311.07742): "Modeling Sequences as Star Graphs to Address Over-smoothing." Cosine similarity among sequence elements rises from 0.5 to 0.8 from block 1 to block 6. Formalizes why deeper SASRec doesn't improve.

## Evaluation methodology

- **Krichene & Rendle, KDD 2020**: "On Sampled Metrics for Item Recommendation." Sampled metrics (100 negatives) are biased and can reverse model rankings. Use full-catalog evaluation when feasible.
- **Petrov & Macdonald, RecSys 2022** (2207.07483): BERT4Rec replicability study (also listed above). Source of this repo's ML-1M preprocessing.
- **Tikhonovich et al., RecSys 2025** (2508.06450): **eSASRec.** Re-benchmarks Transformer recommenders, including HSTU and FuXi-α, on one shared split and finds that HSTU's preprocessing reorders tied timestamps, changing about 25% of the ML-20M test set. Our ML-20M protocol and published comparisons follow this paper. eSASRec itself is SASRec with sampled softmax and a gated Transformer layer (LiGR). Its 8-layer SASRec baseline (HR@10 0.0465 on ML-20M) runs on RecTools' port of the SASRec authors' code; our run of that wiring lands at 0.0446, and our pre-LN block at 0.3196 (CURRICULUM v5).
- **Rendle, Zhang & Koren, 2019** (1905.01395): "On the Difficulty of Evaluating Baselines." Carefully tuned vanilla MF beats most newly proposed methods. The reason this repo insists on getting implementation details right before claiming improvements.

## The bridge to HSTU (2022-2024)

- **Li, Wang & McAuley, WSDM 2020** (doi: 10.1145/3336191.3371786): **TiSASRec.** Time Interval Aware Self-Attention for Sequential Recommendation. Models absolute positions and the time intervals between interactions, not just item order. Our v7's temporal bias buckets draw from this idea.
- **Pancha et al., KDD 2022** (2205.04507): **PinnerFormer.** Pinterest's industrial SASRec-style model. Introduces "Dense All Action" loss (predict all future actions, not just the next item). Pre-LN residuals, sampled softmax.
- **Guo, Wang et al., arXiv 2024** (2412.00714): "Scaling New Frontiers." Compares HSTU, Llama, GPT, and SASRec by varying depth. Table 8 shows SASRec collapses at 4+ blocks on ML-20M. Llama-style residuals + relative attention bias restore scaling. Our v5 sweep does not see the collapse with pre-LN blocks: plain SASRec keeps improving up to 16 blocks, with diminishing returns. With the SASRec authors' released wiring, it collapses from 8 blocks and ties at 4 (CURRICULUM v5). They describe their SASRec as applying the residual "after layer normalization", which matches the authors' code. They train with BCE and 1 negative, which we haven't tried, so we can't say whether that explains their collapse at 4.
- **Zhai et al., ICML 2024** (2402.17152): **HSTU / Generative Recommenders.** "Actions Speak Louder than Words." Replaces softmax attention with SiLU, adds relative attention bias (position + timestamp), adds U(X) gating. M-FALCON inference. Trillion-parameter scale with power-law scaling. The north star.

## HSTU extensions and related architectures

- **FuXi-α, WWW 2025** (2502.03036): Extends the HSTU block with a Modified Feed-Forward Network (MFFN) using SwiGLU in a two-stage architecture. Claims further improvements on top of HSTU's gating mechanism.

*Everything above maps directly to code in this repo or informs its design decisions. Everything below is context on where the field went after HSTU. Useful background, but not implemented here.*

## Future directions

These are techniques from the HSTU paper and related work that are not yet implemented but represent natural next steps for this repo.

- **Stochastic Length Training** (Zhai et al., ICML 2024): During training, randomly sample a contiguous subsequence of each user's history instead of always using the most recent max_len items. This lets the model see different windows of user behavior across epochs, learning both recent and historical patterns without increasing sequence length or memory. At Meta's scale with sequences of 8,192+ items, this is critical for making long-sequence training tractable. On ML-20M with sequences averaging 144 items it may not move metrics, but it is worth understanding.

## Where the field went next (not implemented here)

No new architecture has displaced transformers in production. The field after HSTU
looks like LLMs after the original Transformer: everyone is scaling and adapting the
same core architecture, but via different paths. The work organizes into four lanes.

Terms used below:
- **DLRM** (Deep Learning Recommendation Model): the production standard before generative recommenders. Embedding tables for sparse features, MLPs, and explicit feature interactions.
- **M-FALCON**: HSTU's inference algorithm (Microbatched-Fast Attention Leveraging Cacheable OperatioNs). It scores many candidate items in one pass by computing the user's history once and caching it.
- **DAU**: daily active users.
- **RQ-VAE** (residual-quantized variational autoencoder): compresses an item's content embedding into a short tuple of codes. Each code quantizes what the previous codes left unexplained.
- **Semantic IDs**: item IDs built from those code tuples, so similar items share prefixes. A generative model can then predict an item code by code, like words in a sentence.
- **DPO** (Direct Preference Optimization, Rafailov et al. 2023): fine-tunes a model on pairs of preferred and rejected outputs, without training a separate reward model.

### HSTU-family production scaling

- **Meta, 2026** (2602.16986): ULTRA-HSTU. Sparse attention, attention truncation, Mixture of Transducers. 5x training scaling, 21x inference scaling over baseline HSTU. Deployed to billions of users.
- **Netflix, 2025** ([tech blog](https://netflixtechblog.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39)): Consolidated hundreds of specialized models into a single transformer foundation model over all user interactions. Interaction tokenization, sparse attention, embedding stabilization. Deployed to 300M+ users. Confirms recommendation scaling laws.
- **Meituan, CIKM 2025** (2505.18654): MTGR. Adapts HSTU to retain DLRM-style cross-features. +1.22% take-away orders in production.
- **ByteDance, RecSys 2025** (2505.04421): LONGER. Global tokens + hybrid attention for 10K+ sequences.
- **Huang et al., KDD 2025** (2505.04180): GenRank. Xiaohongshu's generative ranking at scale.

### Semantic ID maturation

- **Rajput et al., NeurIPS 2023** (2305.05065): TIGER. Generative retrieval with Semantic IDs via RQ-VAE. The paper that started this lane.
- **Ye et al., KDD 2024** (2406.14017): EAGER. Two-stream behavioral + semantic SID generation (Huawei).
- **Google/YouTube, 2025** (2510.07784): PLUM. Adapts Gemini-family LLMs with multi-modal Semantic IDs. MoE-3B scale, +5% CTR on YouTube Shorts, launched to billions of users. The biggest production validation of Semantic IDs.
- **Snap Research, CIKM 2025** (2507.22224): GRID. Practitioner's handbook for SID-based generative recommendation. Systematic ablations found RQ-KMeans often matches neural RQ-VAE. Open-source.
- **Zheng et al., Meta, 2025** (2504.02137): Semantic ID prefix-ngram. Deployed in Meta Ads ranking for 1+ years. Prefix-ngram parameterization is key to effectiveness. 0.15% online gain, 43% reduction in A/A prediction variance. Shows SIDs working *within* HSTU/DLRM ranking, bridging the two lanes.
- **Spotify Research, 2025** ([blog](https://research.atspotify.com/2025/9/semantic-ids-for-generative-search-and-recommendation)): Semantic IDs for joint search and recommendation. Key finding: task-specific SIDs don't generalize across tasks, multi-task embeddings needed.
- **Petrov & Macdonald, WSDM 2024** (2312.06165): RecJPQ. Sub-item ID decomposition via product quantization for 47x memory reduction. A different approach to the same item representation scaling problem SIDs address.

### Alignment as post-training

- **Deng et al., 2025** (2502.18965): OneRec. End-to-end generative recommendation with DPO alignment (Kuaishou). Deployed to hundreds of millions of DAU.
- **Tencent, 2025** (2511.10138): GPR. Unified generative advertising on Weixin Channels with Hierarchy Enhanced Policy Optimization (HEPO).
- **JD.com, WWW 2025**: GRAM. Generative retrieval with DPO for e-commerce.
- Most deployed systems (PLUM, PinRec, MTGR, COBRA) still skip alignment. DPO is an emerging pattern, not yet standard.

The only non-transformer contender is Mamba/SSM-based sequential recommendation (Mamba4Rec, SSD4Rec), which offers linear-time complexity but has no production deployments as of early 2026. Pre-HSTU attention variants like BSARec (frequency-domain filtering) and Rec-Denoiser (learned sparse masks) explored fixes to SASRec's limitations but were superseded by the HSTU approach.

### Open-source

- **Meta**: [generative-recommenders](https://github.com/meta-recsys/generative-recommenders). Official HSTU + M-FALCON implementation, and the reference for our HSTU versions.
- **Snap**: [GRID](https://github.com/snap-research/GRID). Modular Semantic ID framework with RQ-KMeans, RQ-VAE, and ablation configs.
- **MiniOneRec** (2510.24431): Open-source SFT to RL workflow for LLM-based generative recommendation.
- **NVIDIA CUTLASS**: Official HSTU attention kernel.
