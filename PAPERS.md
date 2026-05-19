# The SASRec Research Lineage

Every design decision in this repo traces back to a specific paper. This document maps code → research.

The story: SASRec replaces RNNs with Transformers → BERT4Rec challenges it and the
community splits → the loss function is identified as the real problem → SASRec's depth-scaling
failure is diagnosed → PinnerFormer bridges to production → HSTU fixes everything and
unlocks trillion-parameter scale.

---

## Foundation (2016–2018)

- **Hidasi et al., ICLR 2016** (1511.06939): GRU4Rec. First RNN for session-based recommendation. The baseline SASRec was built to replace.
- **Vaswani et al., NeurIPS 2017** (1706.03762): "Attention Is All You Need." The Transformer architecture. SASRec adapts the decoder-only (causal) variant.
- **Kang & McAuley, ICDM 2018** (1808.09781): **SASRec.** Causal self-attention over item sequences for next-item prediction. Trained with BCE loss and 1 negative sample. The architecture this entire repo is built around.
- **Sun et al., CIKM 2019** (1904.06690): **BERT4Rec.** Bidirectional attention with masked-item prediction (Cloze objective). Claimed superiority over SASRec - setting up the debate resolved below.

## The SASRec vs. BERT4Rec debate (2019–2022)

- **Petrov & Macdonald, RecSys 2022** (2207.07483): BERT4Rec replicability study. Reviewed 370 citing papers, found results inconsistent. BERT4Rec wins only 64% of comparisons. Conclusion: BERT4Rec is SOTA *if trained long enough*, but most papers undertrained it. Established the preprocessing and evaluation standards this repo follows.

## The loss function is the real problem (2023–2025)

Two independent groups reached the same conclusion in the same year: SASRec's
underperformance is caused by its loss function, not its architecture.

- **Klenitskiy & Vasilev, RecSys 2023** (2309.07602): "Turning Dross Into Gold Loss." Shows that SASRec trained with cross-entropy loss (full or sampled softmax) matches or exceeds BERT4Rec while training faster. The paper that proved architecture wasn't the bottleneck.
- **Petrov & Macdonald, RecSys 2023** (2308.07192): **gSASRec.** Defines overconfidence formally: BCE + negative sampling causes predicted probabilities to approach 1.0. Proposes gBCE loss with calibration parameter β to fix it. **Best Full Paper Award, RecSys 2023.**
- **Petrov & Macdonald, ACM TORS 2025** (doi:10.1145/3699521): Expanded journal version of gSASRec. Full proofs, calibration error drops from 0.966 to 0.01. Recommends t ∈ [0.75, 0.9] with 128–256 negatives.
- **Jean et al., ACL-IJCNLP 2015** (1412.2007): Original Sampled Softmax for NLP (large vocabulary neural machine translation). The technique gSASRec and the "Dross" paper adapt for recommendation.
- **Zhai et al., KDD 2023** (2306.04039): "Revisiting Neural Retrieval on Accelerators." Applies sampled softmax to large-scale retrieval. Same first author as HSTU. Meta's SASRec implementation uses this loss.

## Implementation details that move metrics 2–5%

- **Huang, arXiv 2025** (2504.09596): "Revisiting Self-Attentive Sequential Recommendation." Positional embedding alignment (steps-back-from-prediction), padding embedding must be frozen at zero, advocacy for native PyTorch attention primitives.
- **D'Ercoli et al., arXiv 2025** (2506.14692): Replicability study of BSARec vs SASRec on a common codebase (EasyRec). Documents that normalization order, dropout placement, and initialization cause 2–5% metric swings.
- **Xiong et al., ICML 2020** (2002.04745): "On Layer Normalization in the Transformer Architecture." The general Pre-LN vs Post-LN analysis.
- **pmixer Result_Norm.md, May 2025**: Pre-LN vs Post-LN empirical results specifically for SASRec. Pre-LN wins on 3/4 datasets.

## Evaluation methodology

- **Krichene & Rendle, KDD 2020**: "On Sampled Metrics for Item Recommendation." Sampled metrics (100 negatives) are biased and can reverse model rankings. Use full-catalog evaluation when feasible.
- **Petrov & Macdonald, RecSys 2022** (2207.07483): BERT4Rec replicability study (also listed above). Established dataset preprocessing standards used throughout this repo.
- **Rendle, Zhang & Koren, 2019** (1905.01395): "On the Difficulty of Evaluating Baselines." Carefully tuned vanilla MF beats most newly proposed methods. The reason this repo insists on getting implementation details right before claiming improvements.

## Why SASRec can't scale deep (2022–2023)

- **Peng et al., 2022** (2209.07997): RAM (Recursive Attentive Methods). Identifies the "localization-deficit": attention distributions approach uniformity in early blocks, item representations diverge from originals. SASRec's optimal depth is stuck at 2 layers.
- **Peng et al., TKDD 2023** (2311.07742): "Modeling Sequences as Star Graphs to Address Over-smoothing." Cosine similarity among sequence elements rises from 0.5 to 0.8 from block 1 to block 6. Formalizes why deeper SASRec doesn't improve.

## The bridge to HSTU (2022–2024)

- **Pancha, Zhai et al., KDD 2022** (2205.04507): **PinnerFormer.** Industrial-scale SASRec at Pinterest. Introduces "Dense All Action" loss (predict all future actions, not just next item). Pre-LN residuals, sampled softmax. Co-authored by Jiaqi Zhai - the direct architectural stepping stone to HSTU.
- **Guo, Wang et al., arXiv 2024** (2412.00714): "Scaling New Frontiers." The Rosetta Stone. Compares HSTU, Llama, GPT, and SASRec by varying depth. Table 8 shows SASRec collapses at 4+ blocks on ML-20M. Llama-style residuals + relative attention bias restore scaling.
- **Zhai et al., ICML 2024** (2402.17152): **HSTU / Generative Recommenders.** "Actions Speak Louder than Words." Replaces softmax attention with SiLU, adds relative attention bias (position + timestamp), adds U(X) gating. M-FALCON inference. Trillion-parameter scale with power-law scaling. The north star.

---

*Everything above maps directly to code in this repo. Everything below is context on where the field went after HSTU - useful background, but not implemented here.*

## Where the field went next (not implemented here)

No new architecture has displaced transformers in production. The field after HSTU
looks like LLMs after the original Transformer - everyone is scaling and adapting the
same core architecture, but via different paths. The work organizes into four lanes.

### HSTU-family production scaling

- **Meta, 2026** (2602.16986): ULTRA-HSTU - sparse attention, attention truncation, Mixture of Transducers. 5× training scaling, 21× inference scaling over baseline HSTU. Deployed to billions of users.
- **Netflix, 2025** ([tech blog](https://netflixtechblog.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39)): Consolidated hundreds of specialized models into a single transformer foundation model over all user interactions. Interaction tokenization, sparse attention, embedding stabilization. Deployed to 300M+ users. Confirms recommendation scaling laws.
- **Meituan, CIKM 2025** (2505.18654): MTGR - adapts HSTU to retain DLRM-style cross-features. +1.22% take-away orders in production.
- **ByteDance, RecSys 2025** (2505.04421): LONGER - global tokens + hybrid attention for 10K+ sequences.
- **Huang et al., KDD 2025** (2505.04180): GenRank - Xiaohongshu's generative ranking at scale.

### Semantic ID maturation

- **Rajput et al., NeurIPS 2023** (2305.05065): TIGER - generative retrieval with Semantic IDs via RQ-VAE. The paper that started this lane.
- **Ye et al., KDD 2024** (2406.14017): EAGER - two-stream behavioral + semantic SID generation (Huawei).
- **Google/YouTube, 2025** (2510.07784): PLUM - adapts Gemini-family LLMs with multi-modal Semantic IDs. MoE-3B scale, +5% CTR on YouTube Shorts, launched to billions of users. The biggest production validation of Semantic IDs.
- **Snap Research, CIKM 2025** (2507.22224): GRID - practitioner's handbook for SID-based generative recommendation. Systematic ablations found RQ-KMeans often matches neural RQ-VAE. Open-source.
- **Zheng et al., Meta, 2025** (2504.02137): Semantic ID prefix-ngram - deployed in Meta Ads ranking for 1+ years. Prefix-ngram parameterization is key to effectiveness. 0.15% online gain, 43% reduction in A/A prediction variance. Shows SIDs working *within* HSTU/DLRM ranking, bridging the two lanes.
- **Spotify Research, 2025** ([blog](https://research.atspotify.com/2025/9/semantic-ids-for-generative-search-and-recommendation)): Semantic IDs for joint search and recommendation. Key finding: task-specific SIDs don't generalize across tasks - multi-task embeddings needed.
- **Petrov & Macdonald, WSDM 2024** (2312.06165): RecJPQ - sub-item ID decomposition via product quantization for 47× memory reduction. A different approach to the same item representation scaling problem SIDs address.

### Alignment as post-training

- **Deng et al., 2025** (2502.18965): OneRec - end-to-end generative recommendation with DPO alignment (Kuaishou). Deployed to hundreds of millions of DAU.
- **Tencent, 2025** (2511.10138): GPR - unified generative advertising on Weixin Channels with Hierarchy Enhanced Policy Optimization (HEPO).
- **JD.com, WWW 2025**: GRAM - generative retrieval with DPO for e-commerce.
- Most deployed systems (PLUM, PinRec, MTGR, COBRA) still skip alignment. DPO is an emerging pattern, not yet standard.

The only non-transformer contender is Mamba/SSM-based sequential recommendation (Mamba4Rec, SSD4Rec), which offers linear-time complexity but has no production deployments as of early 2026. Pre-HSTU attention variants like BSARec (frequency-domain filtering) and Rec-Denoiser (learned sparse masks) explored fixes to SASRec's limitations but were superseded by the HSTU approach.

### Open-source

- **Meta**: [generative-recommenders](https://github.com/meta-recsys/generative-recommenders) - official HSTU + M-FALCON implementation. The definitive reference.
- **Snap**: [GRID](https://github.com/snap-research/GRID) - modular Semantic ID framework with RQ-KMeans, RQ-VAE, and ablation configs.
- **MiniOneRec** (2510.24431): open-source SFT→RL workflow for LLM-based generative recommendation.
- **NVIDIA CUTLASS**: official HSTU attention kernel.