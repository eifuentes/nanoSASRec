# nanoSASRec Walkthrough

Curriculum walkthrough from v1 to v10. The reported results below use the fixed
MovieLens curriculum datasets so they can be compared with published baselines;
the same training scripts also accept prepared KuaiRand outputs.

Each step comes with commands to reproduce the results. Change `--device` to `cpu`, `mps`, or `cuda` based on the hardware available to you.

## Differences

Here are known differences from the original SASRec implementation.

- **Pre-LN normalization** (norm before attention/FFN). The original SASRec uses Post-LN (norm after residual add). Pre-LN is more stable for deeper models (Xiong et al. ICML 2020) and is what all modern implementations use, including HSTU and PinnerFormer.
- **Weight initialization** uses truncated normal with `std=1/sqrt(d)` (matching Zhai et al. (2024)) instead of the original's Xavier uniform. Keeps initial dot products near zero for stable softmax in early training.
- **Reversed positional embeddings** (Huang 2025, arxiv: 2504.09596). The original uses forward indexing; v4+ counts backward from the prediction position so the model learns "how far back" rather than "what absolute slot."

These are intentional refinements, not bugs. v1-v3 use PyTorch default initialization and forward positions to match the original paper; v4+ applies these improvements.

## Dataset Selection

Every Python training script accepts `--dataset` and `--data-dir`. The official walkthrough uses `ml-1m` for v1-v4 and `ml-20m` for v5-v10. Prepare another supported dataset first, then pass its output directory to any compatible training script. The `train_v5.sh` convenience wrapper is fixed to the ML-20M depth sweep.

For KuaiRand, preprocessing creates a recipe-stamped output directory. Keep that complete name when training:

```bash
uv run python data.py --dataset kuairand-pure --kuairand-item-floor 1 --kuairand-user-floor 5
uv run python train_v9.py --dataset kuairand-pure-item1-user5 --device cuda
```

The 27K recipe supports explicit interaction rules, NULL-safe random-impression filtering, DuckDB/Parquet reuse, and analysis-only diagnostics. See the [README dataset guide](README.md#datasets) for the complete Pure and 27K commands.

The examples and metric tables in this walkthrough are MovieLens-specific unless a section says otherwise. KuaiRand catalog sizes, sequence lengths, filtering recipes, and user behavior differ substantially, so KuaiRand results are validation experiments rather than replacements for the published ML-1M/ML-20M comparisons.

## Reading the Metrics

**Avg loss (Sampled Softmax, v2/v4+):** Each training step, the model sees 1 correct item and N random wrong items (256 for ML-1M, 512 for ML-20M in the reported runs) and has to pick the right one. The loss measures how badly it fails at this. If the model has no idea and guesses uniformly across 257 candidates, the probability of guessing right is 1/257 and the loss is -log(1/257) ≈ 5.55. As the model learns, it assigns higher probability to the correct item and the loss drops. The `~X% correct` in the training output is just that probability: how often the model's top pick is actually right. For a different dataset, the relevant baseline changes with `num_negatives` and catalog size.

v2 and v4-v7 write sampled softmax in its explicit `exp/sum/log` form because it mirrors the equation directly. That version can overflow for very large logits, so v8-v10 switch to the stable `logsumexp` form of the same objective. The training objective does not change, only the numerical implementation does.

| Loss | ~P(correct) | What's happening |
|------|-------------|------------------|
| 5.5 | 0.4% | Guessing randomly |
| 3.0 | 5% | Starting to pick up signal |
| 2.0 | 14% | Learning real patterns |
| 1.0 | 37% | Strong preferences |

**Avg loss (BCE, v1/v3):** Instead of picking from a set, the model answers a yes/no question for each item independently: "is this the next item?" A coin flip gives you 50/50, which maps to a loss of -log(0.5) ≈ 0.69. Anything below that means the model is doing better than chance. Not directly comparable to SSM loss values since they measure different things.

**Validation metrics (during training):** HR@10 and NDCG@10 measured against 100 sampled negatives (1 ground truth + 100 random items). Fast enough to run every 10-20 epochs and gives you directional signal: is the model improving? But the absolute numbers are inflated. An HR@10 of 0.95 sounds amazing until you realize ranking 1 item out of 101 is much easier than ranking 1 out of 18,345.

**Test metrics (final benchmark):** HR@10 and NDCG@10 measured against the full item catalog. The model ranks the ground truth against every item in the prepared dataset: 3,416 items for ML-1M, 18,345 for ML-20M, and the recipe-specific catalog for KuaiRand. This is the number you compare against published papers only when the preprocessing and evaluation protocol match. v1-v3 report sampled eval only (they're teaching loss functions, not chasing benchmarks). v4+ reports full-catalog test metrics.

**HR@10 (Hit Rate):** Did the ground truth item land in the top 10? Yes = 1, no = 0, averaged across all users.

**NDCG@10 (Normalized Discounted Cumulative Gain):** Not just "did it make the top 10?" but "how high?" The score uses a log discount: 1/log2(rank+1). Rank 1 gets full credit (1.0), rank 2 gets 0.63, rank 5 gets 0.39, rank 10 gets 0.30. The log means there's a big difference between rank 1 and rank 3, but a small difference between rank 8 and rank 10. This matches real user behavior: people click the first few results, rarely scroll to position 10. When you see an average NDCG@10 of 0.20 across 138K users, it means the typical user's ground truth item lands around rank 4-5 when it does appear in the top 10. NDCG is the metric that matters most for recommendations because getting the right item to the very top is what drives clicks.

## v1: Vanilla SASRec + BCE Loss

### Run

`uv run python train_v1.py --device mps --epochs 200 --eval-every 20`

### Output

```bash
Number Users: 6,040
Number Items: 3,416
Dataset Lengths: 6040 6040 6040
Model parameters: 211,500
Estimate batches per epoch: 47
[03:39:25] [Epoch 1/200] Avg loss: 5.2743 | Avg P(pos): 0.5329
[03:39:27] [Epoch 2/200] Avg loss: 3.8044 | Avg P(pos): 0.5687
[03:39:28] [Epoch 3/200] Avg loss: 2.7431 | Avg P(pos): 0.5634
[03:39:30] [Epoch 4/200] Avg loss: 2.1077 | Avg P(pos): 0.5583
[03:39:32] [Epoch 5/200] Avg loss: 1.7435 | Avg P(pos): 0.5584
[03:39:34] [Epoch 6/200] Avg loss: 1.5299 | Avg P(pos): 0.5618
[03:39:36] [Epoch 7/200] Avg loss: 1.4038 | Avg P(pos): 0.5677
[03:39:37] [Epoch 8/200] Avg loss: 1.3185 | Avg P(pos): 0.5734
[03:39:39] [Epoch 9/200] Avg loss: 1.2701 | Avg P(pos): 0.5787
[03:39:41] [Epoch 10/200] Avg loss: 1.2310 | Avg P(pos): 0.5847
[03:39:43] [Epoch 11/200] Avg loss: 1.1981 | Avg P(pos): 0.5891
[03:39:44] [Epoch 12/200] Avg loss: 1.1731 | Avg P(pos): 0.5896
[03:39:46] [Epoch 13/200] Avg loss: 1.1651 | Avg P(pos): 0.5898
[03:39:48] [Epoch 14/200] Avg loss: 1.1383 | Avg P(pos): 0.5939
[03:39:50] [Epoch 15/200] Avg loss: 1.1221 | Avg P(pos): 0.5986
[03:39:52] [Epoch 16/200] Avg loss: 1.1081 | Avg P(pos): 0.6011
[03:39:54] [Epoch 17/200] Avg loss: 1.0927 | Avg P(pos): 0.6040
[03:39:55] [Epoch 18/200] Avg loss: 1.0868 | Avg P(pos): 0.6060
[03:39:57] [Epoch 19/200] Avg loss: 1.0752 | Avg P(pos): 0.6119
[03:40:07] [Epoch 20/200] Avg loss: 1.0641 | Avg P(pos): 0.6203 | HR@10: 0.4396 NDCG@10: 0.2417
[03:40:52] [Epoch 40/200] Avg loss: 0.9954 | Avg P(pos): 0.6032 | HR@10: 0.4454 NDCG@10: 0.2419
[03:41:38] [Epoch 60/200] Avg loss: 0.8853 | Avg P(pos): 0.6050 | HR@10: 0.5399 NDCG@10: 0.3020
[03:42:24] [Epoch 80/200] Avg loss: 0.8148 | Avg P(pos): 0.6111 | HR@10: 0.6013 NDCG@10: 0.3481
[03:43:10] [Epoch 100/200] Avg loss: 0.7498 | Avg P(pos): 0.6140 | HR@10: 0.6627 NDCG@10: 0.3932
[03:43:57] [Epoch 120/200] Avg loss: 0.7109 | Avg P(pos): 0.6134 | HR@10: 0.6919 NDCG@10: 0.4178
[03:44:42] [Epoch 140/200] Avg loss: 0.6713 | Avg P(pos): 0.6269 | HR@10: 0.7139 NDCG@10: 0.4437
[03:45:28] [Epoch 160/200] Avg loss: 0.6383 | Avg P(pos): 0.6281 | HR@10: 0.7397 NDCG@10: 0.4668
[03:46:13] [Epoch 180/200] Avg loss: 0.6164 | Avg P(pos): 0.6375 | HR@10: 0.7568 NDCG@10: 0.4877
[03:46:58] [Epoch 200/200] Avg loss: 0.6040 | Avg P(pos): 0.6405 | HR@10: 0.7661 NDCG@10: 0.4979
[03:47:07] [Epochs 200] Test metrics: HR@10: 0.7356 NDCG@10: 0.4737

  Prediction demo (user 1):
    Recent history:  Toy Story (1995), Bug's Life, A (1998), Antz (1998), Hunchback of Notre Dame, The (1996), Hercules (1997)
    Ground truth:    Pocahontas (1995)
    Model top-10:
       1. Lady and the Tramp (1955)
       2. Lion King, The (1994)
       3. Little Mermaid, The (1989)
       4. Babe (1995)
       5. Shakespeare in Love (1998)
       6. Jungle Book, The (1967)
       7. Fantasia (1940)
       8. Robin Hood (1973)
       9. Groundhog Day (1993)
      10. West Side Story (1961)
```

### Takeaway

**What we built:** A causal Transformer decoder (SASRec) trained with binary cross-entropy loss and 1 negative per positive, evaluated with leave-one-out ranking against 100 random negatives.

**What we learned about the architecture:** Self-attention lets each position in a user's history decide which past items are most relevant for predicting what comes next. The shifted-sequence trick gives us one training signal per position, not just one per user. Residual connections and layer normalization keep gradients flowing through stacked blocks.

**What the metrics mean:** HR@10 = 0.77 means for 77% of users, the model ranked the actual next item within the top 10 out of 101 candidates. NDCG@10 = 0.51 means that when it did rank correctly, the item tended to land around position 2-3 rather than position 10.

**What's wrong - the overconfidence problem:** The loss dropped to 0.59, which sounds like "almost solved." But map that through the BCE loss curve: a loss of 0.59 means the model assigns P ≈ 0.95 to positive items on average. Average P(pos) across all positions is ~0.64 for v1, which looks reasonable. But the overconfidence lives in the tail: at the final position where predictions matter most, the top 25 items all score above 0.99. The average hides the problem that the ranking metrics expose. When everything is "99% likely," the model can't distinguish between rank 1 and rank 25 - they all look the same. This is overconfidence.

**What 0.99 actually means:** The model is saying "I am 99% sure this is the right next item" for dozens of items simultaneously. That's mathematically incoherent - they can't all be 99% likely. In a catalog of 3,416 items, the true probability of any single item being next is roughly 1/3,416 ≈ 0.03%. A well-calibrated model should predict something close to that for most items and maybe 5-20% for the top few. Predicting 0.99 is overconfident by a factor of ~3,000×.

**Why this happens with BCE:** BCE treats each item as an independent binary question - "is this item next? yes/no." With only 1 negative, the model easily distinguishes positive from negative and pushes its confidence to extremes. It never has to answer the harder question: "which of these items is most likely next?" That competition between candidates is what v2 introduces.

---

## v2: Sampled Softmax Loss

### Run

`uv run python train_v2.py --device mps --epochs 200 --eval-every 20`

### Output

```bash
Number Users: 6,040
Number Items: 3,416
Dataset Lengths: 6040 6040 6040
Model parameters: 211,500
Estimate batches per epoch: 47
[04:10:27] [Epoch 1/200] Avg loss: 18.3133
[04:10:29] [Epoch 2/200] Avg loss: 12.3495
[04:10:30] [Epoch 3/200] Avg loss: 9.2268
[04:10:32] [Epoch 4/200] Avg loss: 7.5412
[04:10:33] [Epoch 5/200] Avg loss: 6.5564
[04:10:35] [Epoch 6/200] Avg loss: 6.0153
[04:10:36] [Epoch 7/200] Avg loss: 5.7089
[04:10:38] [Epoch 8/200] Avg loss: 5.5148
[04:10:39] [Epoch 9/200] Avg loss: 5.3725
[04:10:41] [Epoch 10/200] Avg loss: 5.2841
[04:10:42] [Epoch 11/200] Avg loss: 5.1792
[04:10:44] [Epoch 12/200] Avg loss: 5.1002
[04:10:45] [Epoch 13/200] Avg loss: 5.0373
[04:10:47] [Epoch 14/200] Avg loss: 4.9745
[04:10:48] [Epoch 15/200] Avg loss: 4.9204
[04:10:50] [Epoch 16/200] Avg loss: 4.8806
[04:10:52] [Epoch 17/200] Avg loss: 4.8400
[04:10:53] [Epoch 18/200] Avg loss: 4.7990
[04:10:55] [Epoch 19/200] Avg loss: 4.7922
[04:11:05] [Epoch 20/200] Avg loss: 4.7616 | HR@10: 0.4366 NDCG@10: 0.2424
[04:11:44] [Epoch 40/200] Avg loss: 4.5558 | HR@10: 0.4656 NDCG@10: 0.2591
[04:12:23] [Epoch 60/200] Avg loss: 4.0385 | HR@10: 0.6321 NDCG@10: 0.3816
[04:13:02] [Epoch 80/200] Avg loss: 3.7089 | HR@10: 0.6990 NDCG@10: 0.4449
[04:13:41] [Epoch 100/200] Avg loss: 3.4710 | HR@10: 0.7361 NDCG@10: 0.4869
[04:14:20] [Epoch 120/200] Avg loss: 3.3255 | HR@10: 0.7565 NDCG@10: 0.5151
[04:14:59] [Epoch 140/200] Avg loss: 3.2366 | HR@10: 0.7684 NDCG@10: 0.5361
[04:15:37] [Epoch 160/200] Avg loss: 3.1603 | HR@10: 0.7775 NDCG@10: 0.5487
[04:16:17] [Epoch 180/200] Avg loss: 3.0888 | HR@10: 0.7866 NDCG@10: 0.5566
[04:16:57] [Epoch 200/200] Avg loss: 3.0405 | HR@10: 0.7922 NDCG@10: 0.5699
[04:17:06] [Epochs 200] Test metrics: HR@10: 0.7672 NDCG@10: 0.5471

  Prediction demo (user 1):
    Recent history:  Toy Story (1995), Bug's Life, A (1998), Antz (1998), Hunchback of Notre Dame, The (1996), Hercules (1997)
    Ground truth:    Pocahontas (1995)
    Model top-10:
       1. Lion King, The (1994)
       2. Prince of Egypt, The (1998)
       3. Rescuers Down Under, The (1990)
       4. Sleeping Beauty (1959)
       5. Jungle Book, The (1967)
       6. Little Mermaid, The (1989)
       7. Pinocchio (1940)
       8. 101 Dalmatians (1961)
       9. Fantasia (1940)
      10. Alice in Wonderland (1951)
```

### Takeaway

**What we changed:** Only the loss function. The model architecture is identical to v1 - same SelfAttention, same AttentionBlock, same SASRec. The diff is in three places: sample_batch returns shared-per-user negatives (batch, num_negs) instead of per-position negatives (batch, seq_len), scoring uses matmul for the all-pairs negative computation, and BCE is replaced with sampled softmax.

**What sampled softmax does differently:** Instead of asking "is this item good? yes/no" independently for each candidate (BCE), sampled softmax asks "which of these 257 items is the correct next one?" The positive item must compete directly against 256 negatives in a single softmax distribution. Probabilities must sum to 1, so the model can't give everything a score of 0.99 - pushing one item up forces others down.

**Why the loss doesn't go to zero:** v1's BCE loss dropped from 5.3 to 0.59 and was still falling. v2's sampled softmax loss dropped from 17.9 to 3.07 and plateaued. This is correct and expected. A random model that assigns equal probability to all 257 candidates would have loss -log(1/257) ≈ 5.55. The model got down to 3.07, well below random but nowhere near zero - unlike v1's BCE which kept pushing toward zero.

**Why NDCG improved more than HR:** HR@10 asks a binary question - did the ground truth land in the top 10? v1 already got this right ~77% of the time. The bigger problem was *where* in the top 10 it landed. v1's overconfident scores meant the top 10 items had nearly identical scores, so the ground truth's position within the top 10 was essentially random. v2's better-separated scores push the ground truth toward rank 1-3 when it does appear, which is exactly what NDCG measures. The +0.054 NDCG gain versus +0.017 HR gain is the signature of reduced overconfidence.

**Why shared negatives work:** Each user's 256 negatives are reused across all 199 positions in their sequence. This works because an unseen item is unseen at every position - there's no position-specific reason to sample different negatives. v2 runs in comparable time to v1 (~6-7 min) despite scoring 256× more candidates.

**What's still wrong:** Sampled softmax with 256 negatives is better than BCE with 1, but it's still an approximation of full-catalog softmax (3,416 items). The model sees 256/3,416 ≈ 7.5% of the catalog per training step. And the Petrov & Macdonald research (2308.07192) showed that even sampled softmax exhibits some overconfidence - the predicted probabilities are still higher than the true item relevance probabilities. v3's gBCE solves this more elegantly: it goes back to BCE's simplicity but fixes the overconfidence directly by reweighting the positive/negative contributions to match the true catalog ratio. The punchline - gBCE with just 1 negative matches or beats SSM with 256 negatives, at the same training cost as v1.

## v3: Generalized BCE Loss

### Run

`uv run python train_v3.py --device mps --epochs 200 --eval-every 20`

### Output

```bash
Number Users: 6,040
Number Items: 3,416
Dataset Lengths: 6040 6040 6040
Model parameters: 211,500
Estimate batches per epoch: 47
[13:56:51] [Epoch 1/200] Avg loss: 2.5824 | Avg P(pos): 0.5341
[13:56:53] [Epoch 2/200] Avg loss: 1.7652 | Avg P(pos): 0.5951
[13:56:55] [Epoch 3/200] Avg loss: 1.2478 | Avg P(pos): 0.6069
[13:56:56] [Epoch 4/200] Avg loss: 0.9600 | Avg P(pos): 0.6135
[13:56:58] [Epoch 5/200] Avg loss: 0.8024 | Avg P(pos): 0.6183
[13:57:00] [Epoch 6/200] Avg loss: 0.7115 | Avg P(pos): 0.6228
[13:57:02] [Epoch 7/200] Avg loss: 0.6615 | Avg P(pos): 0.6238
[13:57:04] [Epoch 8/200] Avg loss: 0.6317 | Avg P(pos): 0.6254
[13:57:06] [Epoch 9/200] Avg loss: 0.6145 | Avg P(pos): 0.6248
[13:57:07] [Epoch 10/200] Avg loss: 0.5944 | Avg P(pos): 0.6262
[13:57:09] [Epoch 11/200] Avg loss: 0.5846 | Avg P(pos): 0.6215
[13:57:11] [Epoch 12/200] Avg loss: 0.5711 | Avg P(pos): 0.6207
[13:57:13] [Epoch 13/200] Avg loss: 0.5625 | Avg P(pos): 0.6182
[13:57:15] [Epoch 14/200] Avg loss: 0.5546 | Avg P(pos): 0.6144
[13:57:17] [Epoch 15/200] Avg loss: 0.5443 | Avg P(pos): 0.6139
[13:57:19] [Epoch 16/200] Avg loss: 0.5357 | Avg P(pos): 0.6145
[13:57:21] [Epoch 17/200] Avg loss: 0.5311 | Avg P(pos): 0.6144
[13:57:23] [Epoch 18/200] Avg loss: 0.5224 | Avg P(pos): 0.6179
[13:57:25] [Epoch 19/200] Avg loss: 0.5197 | Avg P(pos): 0.6178
[13:57:35] [Epoch 20/200] Avg loss: 0.5169 | Avg P(pos): 0.6195 | HR@10: 0.4391 NDCG@10: 0.2410
[13:58:19] [Epoch 40/200] Avg loss: 0.4773 | Avg P(pos): 0.6448 | HR@10: 0.4623 NDCG@10: 0.2557
[13:59:03] [Epoch 60/200] Avg loss: 0.4228 | Avg P(pos): 0.6717 | HR@10: 0.5616 NDCG@10: 0.3209
[13:59:47] [Epoch 80/200] Avg loss: 0.3787 | Avg P(pos): 0.6594 | HR@10: 0.6513 NDCG@10: 0.3895
[14:00:31] [Epoch 100/200] Avg loss: 0.3452 | Avg P(pos): 0.6703 | HR@10: 0.7048 NDCG@10: 0.4339
[14:01:17] [Epoch 120/200] Avg loss: 0.3219 | Avg P(pos): 0.6806 | HR@10: 0.7439 NDCG@10: 0.4697
[14:02:02] [Epoch 140/200] Avg loss: 0.3041 | Avg P(pos): 0.6968 | HR@10: 0.7641 NDCG@10: 0.4890
[14:02:48] [Epoch 160/200] Avg loss: 0.2957 | Avg P(pos): 0.7060 | HR@10: 0.7720 NDCG@10: 0.5051
[14:03:36] [Epoch 180/200] Avg loss: 0.2861 | Avg P(pos): 0.7138 | HR@10: 0.7844 NDCG@10: 0.5205
[14:04:25] [Epoch 200/200] Avg loss: 0.2776 | Avg P(pos): 0.7325 | HR@10: 0.7932 NDCG@10: 0.5309
[14:04:34] [Epochs 200] Test metrics: HR@10: 0.7603 NDCG@10: 0.5055

  Prediction demo (user 1):
    Recent history:  Toy Story (1995), Bug's Life, A (1998), Antz (1998), Hunchback of Notre Dame, The (1996), Hercules (1997)
    Ground truth:    Pocahontas (1995)
    Model top-10:
       1. Lion King, The (1994)
       2. Nightmare Before Christmas, The (1993)
       3. Mask, The (1994)
       4. Babe (1995)
       5. Fantasia (1940)
       6. Groundhog Day (1993)
       7. Jungle Book, The (1967)
       8. Men in Black (1997)
       9. Anastasia (1997)
      10. South Park: Bigger, Longer and Uncut (1999)
```

### Takeaway

**What we changed:** Only the loss function, again. The model architecture is identical to v1 and v2. The diff from v2 is in the loss computation: sampled softmax is replaced with generalized binary cross-entropy (gBCE), which goes back to BCE's independent per-item scoring but adds a calibration parameter `t` that downweights the positive signal to correct for undersampling negatives relative to the full catalog.

**What gBCE does differently:** Standard BCE with negative sampling sees a 1:1 positive-to-negative ratio in training, but the real catalog ratio is 1:3,415. The model learns that items are far more likely to be relevant than they actually are. gBCE introduces `t` (0 < t < 1) that scales the positive loss term: `loss = t * (-log(P_pos)) + (1-t) * (-log(1-P_neg))`. This dampens the "push toward 1" gradient while keeping the positive and negative terms balanced, preventing the model from becoming overconfident without starving it of learning signal.

**P(pos) tells the story:** v1's P(pos) climbed to 0.999, making the top 25 items indistinguishable. v3's P(pos) stabilized around 0.73, well below the overconfidence zone but high enough that the model is still learning meaningful preferences. This is the calibration the paper proved theoretically (Theorem 5.1, Petrov & Macdonald 2308.07192): with the right `t`, the predicted probabilities reflect actual item relevance rather than training artifact.

**The interaction between `t` and `num_negatives`:** We ran v3 twice. With 1 negative and `t=0.5`, the model barely learned (HR@10 = 0.44, worse than v1). The positive signal was dampened too aggressively with too little negative information to compensate. With 256 negatives and `t=0.5`, the model matched v2's HR@10 (0.79) while maintaining calibrated probabilities. The lesson: `t` and `num_negatives` are not independent knobs. Fewer negatives need less dampening (higher `t`), more negatives can tolerate stronger dampening (lower `t`). The paper's formula `t = 1/(1 + num_negs * (num_items-1)/num_items)` captures this relationship precisely.

**Why NDCG was lower than v2:** v3's NDCG@10 (0.53) came in below v2's (0.57) despite matching on HR. gBCE's calibrated probabilities spread scores more evenly, which helps avoid overconfidence but doesn't force the hard competition that softmax creates. Sampled softmax explicitly makes items fight for probability mass, producing sharper separation at the top of the ranking. gBCE trades some ranking sharpness for better-calibrated probability estimates. Which matters more depends on the application: if you need probability estimates (e.g., multiplying P(purchase) by item price for revenue prediction), gBCE wins. If you only care about ranking, SSM's sharper top-K separation gives better NDCG.

**What we learned about the loss function progression:** After three versions with identical architecture, the conclusion is clear: SASRec's performance issues were never about the architecture. v1's BCE overconfidence, v2's softmax fix, v3's calibration approach are all statements about how the training signal is shaped, not about how attention works. The architecture was fine from the start. This matches the central finding of the 2023 research: fix the loss, fix the model.

## v4: Implementation Refinements

### Run

`uv run python train_v4.py --device mps`

### Takeaway

**What changed:** Same sampled softmax loss as v2. Five implementation refinements, model decomposed into shared `sasrec/` package.

**Results on ML-1M (200 epochs, dim=50, blocks=2, heads=1, seed=42):**

| Metric | v2 (SSM) | v4 (SSM + refinements) |
|--------|----------|------------------------|
| Val HR@10 | 0.7922 | 0.8490 |
| Val NDCG@10 | 0.5699 | 0.6392 |
| Test HR@10 (full-catalog) | — | 0.3018 |
| Test NDCG@10 (full-catalog) | — | 0.1730 |

Look at the gap between validation and test. Val HR is 0.85 (ranking against 101 items), test HR is 0.30 (ranking against all 3,416 items). Same model, same predictions, wildly different numbers. This is why full-catalog eval matters: sampled eval makes everything look good.

For reference, Zhai et al. (2024) benchmarked SASRec on ML-1M with full-catalog eval and got HR@10 of 0.2852 and NDCG@10 of 0.1603. Our v4 beats that by +5.8% HR despite using the same architecture. The difference is entirely the refinements below.

**What the refinements do and why they matter:**

- **Truncated normal weight init** (`std=1/sqrt(d)`) was the biggest contributor. PyTorch initializes embeddings from N(0,1) by default, meaning each element has variance 1. When you dot product two dim=50 vectors where each element has variance 1, the result has variance equal to the sum: 50. That means initial attention scores land around ±7 (sqrt(50)), which completely saturates softmax before the model has learned anything. Truncated normal with `std=1/sqrt(d)` keeps initial dot products near zero so softmax starts in its useful range. v2 started at loss 18.3; v4 started at 5.4 and reached v2's final HR by epoch 40.
- **Reversed positional embeddings** give the prediction position a consistent embedding (always index 0) instead of varying by sequence length. The model learns "how far back" rather than "what absolute slot." (Huang 2025, arxiv: 2504.09596)
- **Attention weight dropout** was missing in v1-v3. The original Transformer paper applies dropout on attention weights after softmax, not just on the output. One line, standard practice.
- **Registered causal mask buffer** avoids recreating the attention mask every forward pass.
- **Configurable FFN** activation and hidden dimension for experimentation.

**The lesson:** v1-v3 proved the loss function matters. v4 proves implementation details matter just as much: +7% HR and +14% NDCG from changes that didn't touch the loss or model capacity. D'Ercoli et al. 2025 documented 2-5% swings from these details individually; they compound.

## v5: Scaling to ML-20M

### Run

`bash train_v5.sh cuda`

### Takeaway

**What changed:** No new code. v5 runs `train_v4.py` at different depths (2, 4, 8, 16 attention blocks) on a much larger dataset. Two things scale up at once: we jump from ML-1M (6K users, 3.4K items) to ML-20M (138K users, 18K items), and from dim=50 to dim=256 with 8 attention heads. This matches the ML-20M benchmark configuration from Guo et al. (2024) and Zhai et al. (2024), so our numbers are directly comparable to theirs.

**Results on ML-20M (50 epochs, dim=256, heads=8, batch=128, seed=42, full-catalog eval):**

| Blocks | Params | Test HR@10 | Test NDCG@10 | Time/epoch |
|--------|--------|------------|--------------|------------|
| 2 | 5.5M | 0.3009 | 0.1760 | ~75s |
| 4 | 6.3M | 0.3149 | 0.1869 | ~120s |
| 8 | 7.9M | 0.3237 | 0.1941 | ~230s |
| 16 | 11.1M | 0.3350 | 0.2026 | ~420s |

**Depth pays off.** Going from 2 to 16 blocks improves HR by +11.3% and NDCG by +15.1%. This is a clear, monotonic improvement across every depth we tested.

**But depth has diminishing returns.** Each doubling of depth costs roughly 2x the training time but delivers similar or less improvement. At 16 blocks you're spending 5.6x the compute of 2 blocks for +11% quality. The natural next question is whether we can get more from the depth we already have rather than just stacking more blocks.

## v6: ReZero Residual Scaling

### Run

`uv run python train_v6.py --device cuda`

### Takeaway

**The problem with stacking blocks.** Each attention block transforms the input and adds its result back to a residual stream that flows through the whole model. With 2 blocks, this works fine. With 8, you have 8 randomly initialized transformations all adding noise to the stream before any of them have learned anything useful. The original item embeddings get buried. Training eventually sorts this out, but early epochs are wasted fighting through noise.

**ReZero's fix is almost embarrassingly simple.** Multiply each block's output by a learnable scalar, initialized to zero. At the start of training, every block is zeroed out, so the residual stream is just clean item embeddings flowing through. As training progresses, each scalar grows from zero, gradually turning on blocks as they learn something worth adding. The model decides for itself how much each block should contribute and when. Total cost: 16 extra parameters for 8 blocks.

**Results on ML-20M (dim=256, heads=8, 50 epochs, seed=42, full-catalog eval):**

| Config | Blocks | Params | Test HR@10 | Test NDCG@10 |
|--------|--------|--------|------------|--------------|
| v5 | 2 | 5.5M | 0.3009 | 0.1760 |
| v5 | 8 | 7.9M | 0.3237 | 0.1941 |
| v5 | 16 | 11.1M | 0.3350 | 0.2026 |
| v6 (ReZero) | 8 | 7.9M | 0.3365 | 0.2020 |

ReZero at 8 blocks matches plain SASRec at 16 blocks (0.3365 vs 0.3350) with 3.2M fewer parameters and half the training time. 16 scalar parameters did what doubling the model couldn't.

## v7: Relative Position & Temporal Bias

### Run

`uv run python train_v7.py --device cuda`

### Takeaway

**What changed.** Replaced absolute position embeddings with two relative biases added directly to attention scores. Instead of each position having its own learned vector ("what does slot 5 mean?"), the model learns "how far apart are these two items?"

Here's what the distance matrix looks like for a 5-item sequence (each entry is query position minus key position):

```
query ↓    item_1  item_2  item_3  item_4  item_5
item_1  [    0      -1      -2      -3      -4   ]
item_2  [    1       0      -1      -2      -3   ]
item_3  [    2       1       0      -1      -2   ]
item_4  [    3       2       1       0      -1   ]
item_5  [    4       3       2       1       0   ]
```

Negative values mean "this item is in the future." Causal masking zeros those out since items can't attend forward. What remains are the positive distances: "this item was 1 step back, that one was 3 steps back."

We do the same with timestamps. If item_2 happened Monday and item_5 happened the following Thursday, the temporal distance is 3 days. Raw distances can be huge (months apart), so we compress them into 32 log-spaced buckets. Buckets 0-15 cover small distances with fine resolution (0, 1, 2, ... 15 apart). Buckets 16-31 cover larger distances logarithmically (16, 24, 36, ... 200+). Each bucket gets a learnable bias per attention head.

**Results on ML-20M (8 blocks, dim=256, heads=8, 50 epochs, seed=42, full-catalog eval):**

| Config | Test HR@10 | Test NDCG@10 |
|--------|------------|--------------|
| v6 (ReZero, absolute pos) | 0.3365 | 0.2020 |
| v7 (ReZero, relative + temporal) | 0.3457 | 0.2085 |

**Relative bias is the single biggest contributor in the repo.** v7 beats v6 by +2.7% HR and +3.2% NDCG. Unlike absolute position embeddings that encode "what slot is this," relative bias encodes "how far apart are these items" which is what actually matters for sequential patterns. The temporal bias adds real signal too, "items consumed in the same session" gets a different learned weight than "items months apart."

The compactness that originally seemed like a liability (256 parameters vs 50,944 for absolute) is actually a strength. Relative bias uses 32 shared buckets across all positions, which acts as a strong inductive bias, the model must learn general distance patterns rather than memorizing per-slot behavior. This regularization effect compounds with ReZero's residual stabilization.

## v8: HSTU

### Run

`uv run python train_v8.py --device cuda`

### Takeaway

**What is HSTU?** HSTU (Hierarchical Sequential Transduction Unit) is the sequential recommendation block introduced by Zhai et al. (2024). It powers recommendations across Facebook, Instagram, and Threads. v8 implements the first local version inside `train_v8.py`; v9+ uses the shared implementation in `sasrec/model/hstu.py`.

Three things change from SASRec's Transformer block:

**1. SiLU replaces softmax in attention.** Softmax normalizes attention weights to sum to 1, making it a zero-sum game: attending more to one item means attending less to others. SiLU removes this constraint. Each item's weight is independent, so the model can strongly attend to several items at once without diluting the signal.

**2. LayerNorm after attention.** Without softmax keeping values bounded, attention outputs can grow large. LayerNorm stabilizes them.

**3. Gating replaces the feed-forward network.** SASRec uses a 2-layer FFN after attention. HSTU replaces it with a pointwise gate: a separate projection U(X) from the raw input gets multiplied element-wise with the attention output. Each feature dimension is the product of "what attention found" and "what the raw input says." This lets the model selectively pass, amplify, or suppress each dimension. Same idea as SwiGLU in LLaMA.

**Results on ML-20M (8 blocks, dim=256, heads=8, 50 epochs, seed=42, full-catalog eval):**

| Config | Test HR@10 | Test NDCG@10 |
|--------|------------|--------------|
| v5 blocks=8 (SASRec) | 0.3237 | 0.1941 |
| v6 (+ ReZero) | 0.3365 | 0.2020 |
| **v7 (+ relative bias)** | **0.3457** | **0.2085** |
| v8 (HSTU) | 0.3339 | 0.2006 |
| SASRec (Zhai et al., 2024) | 0.2889 | 0.1621 |
| HSTU (Zhai et al., 2024) | 0.3273 | 0.1895 |
| HSTU-large (Zhai et al., 2024) | 0.3556 | 0.2098 |

**The architecture swap is not a clear win at this scale.** v8 HSTU (0.3339) beats the v5 8-block SASRec baseline (0.3237), but lands below v6 ReZero (0.3365) and v7 relative bias (0.3457). The HSTU block replaces softmax+FFN with SiLU+gating while keeping the same relative bias infrastructure from v7.

**The bigger finding.** Our v7 SASRec+ReZero+relative bias (0.3457) beats the Zhai et al. (2024) HSTU result (0.3273) by +5.6% without changing the attention mechanism. Training improvements and proper relative bias implementation explain most of the gap between "old SASRec" and "HSTU" in the published benchmark.

## v9: φ1 SwishLayerNorm Pre-Activation

### Run

`uv run python train_v9.py --device cuda`

### Takeaway

**What changed.** Combined the four separate Q, K, V, U linear projections into a single projection (dim to 4*dim), then applied SwishLayerNorm before splitting them apart. SwishLayerNorm is `x * sigmoid(LayerNorm(x))`: LayerNorm normalizes the combined projection, sigmoid gates each dimension between 0 and 1, and multiplying by the original lets the model decide how much to keep.

This matters because the projections must be combined. LayerNorm computes statistics (mean, variance) across the full 4*dim dimension, so Q's normalization depends on what U, V, and K look like. Separate projections with separate LayerNorms would be a different operation.

The result is that Q, K, V, U become nonlinear functions of the input before they interact. In v8, attention scores were a bilinear form (linear Q dotted with linear K). In v9, both Q and K pass through a nonlinearity first, giving the model a richer set of patterns it can represent.

**Results on ML-20M (8 blocks, dim=256, heads=8, 50 epochs, seed=42, full-catalog eval):**

| Config | Params | Test HR@10 | Test NDCG@10 |
|--------|--------|------------|--------------|
| v8 (HSTU, separate projections) | 7.3M | 0.3339 | 0.2006 |
| **v9 (HSTU, combined + φ1)** | **7.3M** | **0.3425** | **0.2071** |
| HSTU (Zhai et al., 2024) | — | 0.3273 | 0.1895 |

**φ1 improved over v8 by +2.6% HR.** v9 beats the Zhai et al. (2024) HSTU result by +4.6% HR and +9.3% NDCG. But v7 SASRec+ReZero+relative bias (0.3457) still leads the repo, reinforcing that training and positional improvements matter more than the attention mechanism at this scale.

## v10: Training Alignment

### Run

`uv run python train_v10.py --device cuda`

### Takeaway

**What changed.** Three training details from the Zhai et al. (2024) HSTU setup, applied on top of v9's architecture:

**L2 normalization.** Normalize both the model's output embeddings and the item embeddings to unit length before scoring. This turns dot products into cosine similarity, bounded between [-1, 1]. Without it, the model can give popular items larger embedding norms so they score high regardless of relevance. With it, all items compete on direction (angular similarity) alone.

**Temperature scaling.** Divide all logits by a temperature parameter (0.05) before the softmax loss. With cosine similarity in [-1, 1], dividing by 0.05 scales logits to [-20, 20]. This forces the softmax to commit hard to its top picks. A higher temperature (softer distribution) lets the model be vague about rankings.

**Collision masking.** If a randomly sampled negative happens to be the same item as the positive target, mask it to negative infinity before the loss. Without this, the model occasionally gets penalized for scoring a correct item highly. With 512 negatives out of 18,345 items, collisions happen about 2.8% of the time per position.

These are standard techniques in production recommendation systems and appear in the Zhai et al. (2024) HSTU training setup.

**Results on ML-20M (8 blocks, dim=256, heads=8, 50 epochs, seed=42, full-catalog eval):**

| Config | HR@10 | NDCG@10 |
|--------|-------|---------|
| v9 (raw dot products) | 0.3425 | 0.2071 |
| v10 temp=0.05 | 0.3355 | 0.2016 |

**These techniques didn't help at our scale.** v10 slightly underperformed v9 across all temperatures tested. The likely reason is the negative sampling strategy: we use random negatives, while Zhai et al. (2024) use in-batch negatives (the positive items from other users in the same batch).

Random negatives are "easy" because they're mostly items from completely different categories. The model quickly learns to reject them and doesn't need temperature sharpening to separate them. In-batch negatives are "hard" because they're items other users actually interacted with, forcing finer distinctions. That's where temperature sharpening and cosine similarity pay off.

L2 normalization also removes a degree of freedom. Without it, the model can use embedding magnitude as a signal (popular items naturally develop larger norms). At production scale with billions of interactions, this prevents unhealthy popularity bias. At ML-20M scale, the popularity signal is useful and removing it hurts.

**The lesson for practitioners.** These training details are not universally beneficial. They're coupled to the full training pipeline, particularly the negative sampling strategy and catalog scale. Adopting the Zhai et al. (2024) scoring setup without adopting in-batch negative mining won't reproduce the same behavior.

## Appendix: Dense All Action

### Run

`uv run python train_v11.py --device cuda`
`uv run python train_v11.py --device cuda --horizon 10`

### Takeaway

**What changed.** Instead of each position predicting only the next item, every position predicts ALL future items in the sequence. Position 3 predicts items 4, 5, 6, ... all the way to the end. This gives roughly 100x more gradient signal per sequence without changing the model.

The implementation is memory-efficient: precompute `logsumexp` of negative scores once per position, then reuse it for every future target. Total extra memory is one (batch, seq_len, seq_len) tensor.

**Results on ML-20M (8 blocks, dim=256, heads=8, 50 epochs, seed=42, full-catalog eval):**

| Variant | HR@10 | NDCG@10 |
|---------|-------|---------|
| v9 (next-item only) | 0.3425 | 0.2071 |
| Dense All Action (K=10 horizon) | 0.2565 | 0.1416 |

**Why it doesn't work for next-item evaluation.** For a sequence of length 199 with horizon K=10, each position predicts its next 10 items. The next-item prediction (the diagonal) is 1 out of 10 targets, so only 10% of the gradient signal targets the metric we actually evaluate on. The other 90% optimizes for predicting items 2-10 steps ahead.

**What Dense All Action is actually for.** PinnerFormer introduced this technique for temporal embedding stability at Pinterest. If the model can predict not just the next item but all future items, then user embeddings computed today are still useful for serving recommendations tomorrow. The evaluation in PinnerFormer reflects this: they measure whether embeddings from 24 hours ago still produce good recommendations today. That's a different objective than next-item HR@10.

**The open-source generative-recommenders implementation does not use Dense All Action for the ML-20M benchmark.** The Zhai et al. (2024) ML-20M results (HR@10 0.3273) use standard next-item sampled softmax loss, the same as our v9. Dense All Action is referenced conceptually but is not part of the benchmark code.

## Applying The Curriculum To KuaiRand

KuaiRand is supported as a separate dataset family rather than as an additional official benchmark row. Its raw logs contain engagement signals and filtering choices that are not present in MovieLens, so the interaction recipe is part of the experiment definition.

Prepare the confirmed Pure slice for a smaller validation run:

```bash
uv run python data.py --dataset kuairand-pure --kuairand-item-floor 1 --kuairand-user-floor 5
uv run python train_v7.py --dataset kuairand-pure-item1-user5 --device cuda --epochs 50
```

Prepare the long-view 27K slice when testing longer histories or explicit engagement filtering:

```bash
uv run python data.py \
  --dataset kuairand-27k \
  --kuairand-interaction-rule long-view-not-hate-tab1 \
  --kuairand-exclude-rand \
  --kuairand-item-floor 555 \
  --kuairand-user-floor 5
uv run python train_v9.py \
  --dataset kuairand-27k-rule-long-view-not-hate-tab1-item555-user5-exclude-rand \
  --device cuda --epochs 200 --eval-every 20
```

Use `--kuairand-analysis-only` to inspect stage counts, tie statistics, and item-frequency profiles before exporting a training split. Each exported recipe records its parameters in `recipe.json`; retain that file with any reported result. The current 27K recipe is a principled long-view engagement slice and is not a reproduction of the FuXi-alpha checksum table.
