# nanoSASRec Walkthrough

Curriculum walkthrough from v1 to v10.

Each step comes with commands to reproduce the results. Change `--device` to `cpu`, `mps`, or `cuda` based on the hardware available to you.

## Differences

Here are known differences from the original SASRec implementation.

- **Pre-LN normalization** (norm before attention/FFN). The original SASRec uses Post-LN (norm after residual add). Pre-LN is more stable for deeper models (Xiong et al. ICML 2020) and is what all modern implementations use, including HSTU and PinnerFormer.
- **Weight initialization** uses truncated normal with `std=1/sqrt(d)` (matching Zhai et al. (2024)) instead of the original's Xavier uniform. Keeps initial dot products near zero for stable softmax in early training.
- **Reversed positional embeddings** (Huang 2025, arxiv: 2504.09596). The original uses forward indexing; v4+ counts backward from the prediction position so the model learns "how far back" rather than "what absolute slot."

These are intentional refinements, not bugs. v1-v3 use PyTorch default initialization and forward positions to match the original paper; v4+ applies these improvements.

## Reading the Metrics

**Avg loss (Sampled Softmax, or SSM, v2/v4+):** Each training step, the model sees 1 correct item and N random wrong items (256 for ML-1M, 512 for ML-20M) and has to pick the right one. The loss measures how badly it fails at this. If the model has no idea and guesses uniformly across 257 candidates, the probability of guessing right is 1/257 and the loss is -log(1/257) ≈ 5.55 (≈ 6.24 with ML-20M's 513 candidates). As the model learns, it assigns higher probability to the correct item and the loss drops. The `~X% correct` in the training output is just that probability: how often the model's top pick is actually right.

v2 and v4-v7 write sampled softmax in its explicit `exp/sum/log` form because it mirrors the equation directly. That version can overflow for very large logits, so v8-v10 switch to the stable `logsumexp` form of the same objective. The training objective does not change, only the numerical implementation does.

| Loss | ~P(correct) | What's happening |
|------|-------------|------------------|
| 5.5 | 0.4% | Guessing randomly |
| 3.0 | 5% | Starting to pick up signal |
| 2.0 | 14% | Learning real patterns |
| 1.0 | 37% | Strong preferences |

**Avg loss (BCE, v1/v3):** Instead of picking from a set, the model answers a yes/no question for each item independently: "is this the next item?" A coin flip gives you 50/50, which maps to a loss of -log(0.5) ≈ 0.69. Anything below that means the model is doing better than chance. Not directly comparable to SSM loss values since they measure different things. v3 averages the loss over the positive and its 256 negatives, so its values are much smaller (around 0.006).

**Validation metrics (during training):** HR@10 and NDCG@10 on the second-to-last item of each user. v1-v3 rank it against 100 sampled negatives (1 ground truth + 100 random items). That's fast, but the absolute numbers are inflated. An HR@10 of 0.95 sounds amazing until you realize ranking 1 item out of 101 is much easier than ranking 1 out of 26,744. Sampled metrics can also change which of two models looks better (Krichene & Rendle, KDD 2020). So v4+ validate on the full catalog, the same way as test. Validation and test then use the same yardstick. On ML-20M, validation HR@10 still runs about 3 points above test, because the last item is consistently harder to predict than the second-to-last. The gap is steady across versions, so validation is a trustworthy signal for picking models and checkpoints.

**Test metrics (final benchmark):** HR@10 and NDCG@10 on the last item of each user, given everything before it, measured against the full item catalog. The model ranks the ground truth against every item in the dataset, except items already in the user's history: 3,416 items for ML-1M and 26,744 for ML-20M. This is the number you compare against published papers. ML-20M uses the same unfiltered catalog as Zhai et al. (2024) and eSASRec (Tikhonovich et al., 2025). v1-v3 report sampled eval only (they're teaching loss functions, not chasing benchmarks). v4+ reports full-catalog test metrics.

**HR@10 (Hit Rate):** Did the ground truth item land in the top 10? Yes = 1, no = 0, averaged across all users.

**NDCG@10 (Normalized Discounted Cumulative Gain):** Not just "did it make the top 10?" but "how high?" The score uses a log discount: 1/log2(rank+1). Rank 1 gets full credit (1.0), rank 2 gets 0.63, rank 5 gets 0.39, rank 10 gets 0.30. The log means there's a big difference between rank 1 and rank 3, but a small difference between rank 8 and rank 10. This matches real user behavior: people click the first few results, rarely scroll to position 10. When you see an average NDCG@10 of 0.20 across 138K users, with HR@10 around 0.34, it means the ground truth item lands around rank 2-3 on average when it does appear in the top 10. NDCG is the metric that matters most for recommendations because getting the right item to the very top is what drives clicks.

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

**What the metrics mean:** HR@10 = 0.77 means for 77% of users, the model ranked the actual next item within the top 10 out of 101 candidates. NDCG@10 = 0.50 means that when it did rank correctly, the item tended to land around position 2-3 rather than position 10.

**What's wrong - the overconfidence problem:** The loss dropped to 0.60, which sounds like "almost solved." But map that through the BCE loss curve: a loss of 0.60 means the model assigns P ≈ 0.95 to positive items on average. Average P(pos) across all positions is ~0.64 for v1, which looks reasonable. But the overconfidence lives in the tail: at the final position where predictions matter most, the top 25 items all score above 0.99. The average hides the problem that the ranking metrics expose. When everything is "99% likely," the model can't distinguish between rank 1 and rank 25 - they all look the same. This is overconfidence.

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

**Why the loss doesn't go to zero:** v1's BCE loss dropped from 5.3 to 0.60 and was still falling. v2's sampled softmax loss dropped from 18.3 to 3.04 and plateaued. This is correct and expected. A random model that assigns equal probability to all 257 candidates would have loss -log(1/257) ≈ 5.55. The model got down to 3.04, well below random but nowhere near zero - unlike v1's BCE which kept pushing toward zero.

**Why NDCG improved more than HR:** HR@10 asks a binary question - did the ground truth land in the top 10? v1 already got this right ~77% of the time. The bigger problem was *where* in the top 10 it landed. v1's overconfident scores meant the top 10 items had nearly identical scores, so the ground truth's position within the top 10 was essentially random. v2's better-separated scores push the ground truth toward rank 1-3 when it does appear, which is exactly what NDCG measures. The +0.072 NDCG gain versus +0.026 HR gain is the signature of reduced overconfidence.

**Why shared negatives work:** Each user's 256 negatives are reused across all 199 positions in their sequence. This works because an unseen item is unseen at every position - there's no position-specific reason to sample different negatives. v2 runs in comparable time to v1 (~6-7 min) despite scoring 256× more candidates.

**What's still wrong:** Sampled softmax with 256 negatives is better than BCE with 1, but it's still an approximation of full-catalog softmax (3,416 items). The model sees 256/3,416 ≈ 7.5% of the catalog per training step. And the Petrov & Macdonald research (2308.07192) showed that even sampled softmax exhibits some overconfidence - the predicted probabilities are still higher than the true item relevance probabilities. v3's gBCE takes a different route: it goes back to BCE and corrects the overconfidence directly, by shrinking the positive term according to how few negatives were sampled.

## v3: Generalized BCE Loss

### Run

`uv run python train_v3.py --device mps --epochs 400 --eval-every 20`

### Output

```bash
Number Users: 6,040
Number Items: 3,416
Dataset Lengths: 6040 6040 6040
Model parameters: 211,500
Estimate batches per epoch: 47
[05:12:34] [Epoch 1/400] Avg loss: 2.4040 | Avg P(pos): 0.4965
[05:12:36] [Epoch 2/400] Avg loss: 1.3705 | Avg P(pos): 0.4623
[05:12:39] [Epoch 3/400] Avg loss: 0.7902 | Avg P(pos): 0.4124
[05:12:41] [Epoch 4/400] Avg loss: 0.4670 | Avg P(pos): 0.3651
[05:12:44] [Epoch 5/400] Avg loss: 0.2810 | Avg P(pos): 0.3302
[05:12:47] [Epoch 6/400] Avg loss: 0.1705 | Avg P(pos): 0.3018
[05:12:50] [Epoch 7/400] Avg loss: 0.1009 | Avg P(pos): 0.2830
[05:12:53] [Epoch 8/400] Avg loss: 0.0566 | Avg P(pos): 0.2741
[05:12:55] [Epoch 9/400] Avg loss: 0.0318 | Avg P(pos): 0.2655
[05:12:58] [Epoch 10/400] Avg loss: 0.0196 | Avg P(pos): 0.2609
[05:13:01] [Epoch 11/400] Avg loss: 0.0140 | Avg P(pos): 0.2643
[05:13:04] [Epoch 12/400] Avg loss: 0.0116 | Avg P(pos): 0.2637
[05:13:06] [Epoch 13/400] Avg loss: 0.0105 | Avg P(pos): 0.2639
[05:13:09] [Epoch 14/400] Avg loss: 0.0100 | Avg P(pos): 0.2740
[05:13:12] [Epoch 15/400] Avg loss: 0.0098 | Avg P(pos): 0.2753
[05:13:15] [Epoch 16/400] Avg loss: 0.0096 | Avg P(pos): 0.2892
[05:13:17] [Epoch 17/400] Avg loss: 0.0094 | Avg P(pos): 0.2894
[05:13:20] [Epoch 18/400] Avg loss: 0.0093 | Avg P(pos): 0.3057
[05:13:23] [Epoch 19/400] Avg loss: 0.0091 | Avg P(pos): 0.3189
[05:13:36] [Epoch 20/400] Avg loss: 0.0090 | Avg P(pos): 0.3313 | HR@10: 0.4320 NDCG@10: 0.2343
[05:14:45] [Epoch 40/400] Avg loss: 0.0082 | Avg P(pos): 0.4462 | HR@10: 0.4379 NDCG@10: 0.2388
[05:15:53] [Epoch 60/400] Avg loss: 0.0082 | Avg P(pos): 0.4426 | HR@10: 0.4399 NDCG@10: 0.2419
[05:16:56] [Epoch 80/400] Avg loss: 0.0081 | Avg P(pos): 0.4404 | HR@10: 0.4444 NDCG@10: 0.2474
[05:18:04] [Epoch 100/400] Avg loss: 0.0079 | Avg P(pos): 0.3918 | HR@10: 0.4914 NDCG@10: 0.2824
[05:19:07] [Epoch 120/400] Avg loss: 0.0077 | Avg P(pos): 0.3235 | HR@10: 0.5561 NDCG@10: 0.3258
[05:20:15] [Epoch 140/400] Avg loss: 0.0075 | Avg P(pos): 0.3237 | HR@10: 0.6071 NDCG@10: 0.3629
[05:21:25] [Epoch 160/400] Avg loss: 0.0073 | Avg P(pos): 0.3891 | HR@10: 0.6354 NDCG@10: 0.3887
[05:22:33] [Epoch 180/400] Avg loss: 0.0072 | Avg P(pos): 0.4355 | HR@10: 0.6629 NDCG@10: 0.4217
[05:23:40] [Epoch 200/400] Avg loss: 0.0070 | Avg P(pos): 0.4540 | HR@10: 0.6935 NDCG@10: 0.4471
[05:24:22] [Epoch 220/400] Avg loss: 0.0069 | Avg P(pos): 0.4659 | HR@10: 0.7061 NDCG@10: 0.4644
[05:25:05] [Epoch 240/400] Avg loss: 0.0068 | Avg P(pos): 0.4695 | HR@10: 0.7243 NDCG@10: 0.4842
[05:25:48] [Epoch 260/400] Avg loss: 0.0067 | Avg P(pos): 0.4770 | HR@10: 0.7407 NDCG@10: 0.5022
[05:26:30] [Epoch 280/400] Avg loss: 0.0066 | Avg P(pos): 0.4857 | HR@10: 0.7488 NDCG@10: 0.5121
[05:27:13] [Epoch 300/400] Avg loss: 0.0065 | Avg P(pos): 0.4665 | HR@10: 0.7573 NDCG@10: 0.5230
[05:27:56] [Epoch 320/400] Avg loss: 0.0065 | Avg P(pos): 0.4749 | HR@10: 0.7618 NDCG@10: 0.5290
[05:28:38] [Epoch 340/400] Avg loss: 0.0064 | Avg P(pos): 0.4861 | HR@10: 0.7685 NDCG@10: 0.5332
[05:29:21] [Epoch 360/400] Avg loss: 0.0064 | Avg P(pos): 0.4771 | HR@10: 0.7752 NDCG@10: 0.5406
[05:30:03] [Epoch 380/400] Avg loss: 0.0063 | Avg P(pos): 0.4735 | HR@10: 0.7732 NDCG@10: 0.5448
[05:30:46] [Epoch 400/400] Avg loss: 0.0063 | Avg P(pos): 0.4734 | HR@10: 0.7790 NDCG@10: 0.5498
[05:30:54] [Epochs 400] Test metrics: HR@10: 0.7596 NDCG@10: 0.5318

  Prediction demo (user 1):
    Recent history:  Toy Story (1995), Bug's Life, A (1998), Antz (1998), Hunchback of Notre Dame, The (1996), Hercules (1997)
    Ground truth:    Pocahontas (1995)
    Model top-10:
       1. Lion King, The (1994)
       2. Little Mermaid, The (1989)
       3. Lady and the Tramp (1955)
       4. Peter Pan (1953)
       5. Jungle Book, The (1967)
       6. Anastasia (1997)
       7. Sleeping Beauty (1959)
       8. Fantasia (1940)
       9. 101 Dalmatians (1961)
      10. Pocahontas (1995) <- hit
```

### Takeaway

**What we changed:** Only the loss function, again. The model architecture is identical to v1 and v2, and v3 samples 256 negatives per user the same way v2 does. The diff from v2 is the loss: sampled softmax is replaced with generalized binary cross-entropy (gBCE). gBCE goes back to BCE's independent yes/no question for each item, then corrects BCE's overconfidence with one calibration parameter, `t`.

**What gBCE does differently:** With 256 sampled negatives, the model sees only α = 256/3,415 ≈ 7.5% of the items it should be ranked against. BCE trained this way learns probabilities that are far too high: it has no idea how many other items it never saw. gBCE fixes this by raising the positive's predicted probability to a power β, which is the same as multiplying the positive term by β: `-log(P^β) = β · -log(P)`. β comes from `t` and the sampling rate: `β = α · (t · (1 - 1/α) + 1/α)`. With `t = 0`, β = 1 and gBCE is plain BCE. With `t = 1`, β = α and the correction is complete. The paper's default, `t = 0.75`, gives β ≈ 0.31 here. As in the paper's reference code, the loss is the average of the positive term and the 256 negative terms.

**gBCE needs more epochs.** v1, v2 and v4 train for 200 epochs. v3 trains for 400, because at 200 it is still far from done:

| Loss (ML-1M, 256 negatives) | Epochs | Val HR@10 | Val NDCG@10 | Test HR@10 | Test NDCG@10 |
|-----------------------------|--------|-----------|-------------|------------|--------------|
| v2: sampled softmax | 200 | 0.7922 | 0.5699 | 0.7672 | 0.5471 |
| plain BCE (`t = 0`, β = 1) | 200 | 0.7270 | 0.4816 | 0.6959 | 0.4586 |
| gBCE (`t = 0.75`, β ≈ 0.31) | 200 | 0.6944 | 0.4485 | 0.6704 | 0.4285 |
| **v3: gBCE (`t = 0.75`)** | **400** | **0.7790** | **0.5498** | **0.7596** | **0.5318** |

The positive is one term out of 257, so the negatives dominate the early gradient. With either BCE variant, validation HR@10 sits near 0.44 for about 80 epochs, which is roughly what recommending popular items gets you, before the model starts using the sequence. (v4's initialization shortens plateaus like this one.) Shrinking the positive term further with β < 1 slows learning again, so at 200 epochs gBCE is behind plain BCE. Given 400 epochs, gBCE reaches sampled softmax on HR@10 (0.7596 against 0.7672) but stays behind on NDCG@10 (0.5318 against 0.5471). The prediction demo shows the difference: at 200 epochs the top 10 is popular 1990s films; at 400 it is Disney animation, with the ground truth at rank 10.

**`t` and the number of negatives work together.** α is the fraction of negatives sampled, so the size of the correction depends on both knobs. With all 3,415 negatives, α = 1 and β = 1 for any `t`: there is nothing to correct. With 256 negatives and `t = 0.75`, β ≈ 0.31. With a single negative, β ≈ 0.25. `num_negatives` sets how big the gap is; `t` sets how much of it to close.

**What calibration buys, and what it doesn't.** gBCE's benefit is in the probability values themselves, which matter when you use a score as a probability, for example multiplying P(purchase) by price to estimate revenue. HR@10 and NDCG@10 only look at the order of the scores, so they can't show it. The `Avg P(pos)` column in the output doesn't show it either: it is the average predicted probability of the true next item during training, and it ends close for gBCE (0.45) and plain BCE (0.48) at 200 epochs. The paper measures calibration against the full catalog, which this repo doesn't compute.

**Why NDCG trails sampled softmax:** gBCE scores each item independently. Sampled softmax makes the candidates compete for probability mass, which pushes the ground truth sharply above its neighbours and helps NDCG most. If you only care about ranking, sampled softmax is the better and faster choice. That is why v4+ use it.

**What we learned about the loss function progression:** After three versions with identical architecture, the conclusion is clear: SASRec's performance issues were never about the architecture. v1's BCE overconfidence, v2's softmax fix, v3's calibration approach are all statements about how the training signal is shaped, not about how attention works. The architecture was fine from the start. This matches the central finding of the 2023 research: fix the loss, fix the model.

## v4: Implementation Refinements

### Run

`uv run python train_v4.py --device mps`

### Takeaway

**What changed:** Same sampled softmax loss as v2. Five implementation refinements, model decomposed into shared `sasrec/` package.

**Results on ML-1M (200 epochs, dim=50, blocks=2, heads=1, seed=42):**

| Metric | v2 (SSM) | v4 (SSM + refinements) |
|--------|----------|------------------------|
| Val HR@10 (sampled, 100 negatives)* | 0.7922 | 0.8490 |
| Val NDCG@10 (sampled, 100 negatives)* | 0.5699 | 0.6392 |
| Val HR@10 (full-catalog) | — | 0.3189 |
| Val NDCG@10 (full-catalog) | — | 0.1760 |
| Test HR@10 (full-catalog) | — | 0.3017 |
| Test NDCG@10 (full-catalog) | — | 0.1718 |

\*v4's sampled validation numbers are from v0.2.1, before v4+ switched to full-catalog validation. They are kept so v2 and v4 can be compared on the same metric.

Look at the gap between the two kinds of validation. Ranked against 101 items, v4's validation HR was 0.85. Ranked against all 3,416 items, it is 0.32, close to the test HR of 0.30. The same v4 recipe, measured two ways, gives wildly different numbers. This is why full-catalog eval matters: sampled eval makes everything look good. It is also why v4+ validate on the full catalog: validation should predict test.

For reference, Zhai et al. (2024) benchmarked SASRec on ML-1M with full-catalog eval and got HR@10 of 0.2852 and NDCG@10 of 0.1603. Our v4 beats that by +5.8% HR despite using the same architecture. The difference is entirely the refinements below.

**What the refinements do and why they matter:**

- **Truncated normal weight init** (`std=1/sqrt(d)`) was the biggest contributor. PyTorch initializes embeddings from N(0,1) by default, meaning each element has variance 1. When you dot product two dim=50 vectors where each element has variance 1, the result has variance equal to the sum: 50. That means initial attention scores land around ±7 (sqrt(50)), which completely saturates softmax before the model has learned anything. Truncated normal with `std=1/sqrt(d)` keeps initial dot products near zero so softmax starts in its useful range. v2 started at loss 18.3; v4 started at 5.4 and reached v2's final HR by epoch 40.
- **Reversed positional embeddings** give the prediction position a consistent embedding (always index 0) instead of varying by sequence length. The model learns "how far back" rather than "what absolute slot." (Huang 2025, arxiv: 2504.09596)
- **Attention weight dropout** was missing in v1-v3. The original Transformer paper applies dropout on attention weights after softmax, not just on the output. One line, standard practice.
- **Registered causal mask buffer** avoids recreating the attention mask every forward pass.
- **Configurable FFN** activation and hidden dimension for experimentation.

**The lesson:** v1-v3 proved the loss function matters. v4 proves implementation details matter just as much: +7% HR and +12% NDCG from changes that didn't touch the loss or model capacity. D'Ercoli et al. 2025 documented 2-5% swings from these details individually; they compound.

## v5: Scaling to ML-20M

### Run

`bash train_v5.sh cuda`

### Takeaway

> **Note on v5-v11 results:** these numbers use the full 26,744-item ML-20M catalog, the same catalog as Zhai et al. (2024) and eSASRec (Tikhonovich et al., 2025). v0.2.1, the previous release, filtered ML-20M to items with at least 5 ratings (18,345 items). Its numbers were 0.4 to 1.0 points of HR@10 higher, because the smaller catalog makes ranking easier, but the 8-block versions ranked in the same order (v7 > v9 > v6 > v10 > v8 > v5). v7 and v9 report the mean ± standard deviation over seeds 42, 43 and 44. The seed-to-seed spread, about 0.1 points of HR@10, is well below the gaps between versions.

**What changed:** No new code. v5 runs `train_v4.py` at different depths (2, 4, 8, 16 attention blocks) on a much larger dataset. Two things scale up at once: we jump from ML-1M (6K users, 3.4K items) to ML-20M (138K users, 27K items), and from dim=50 to dim=256 with 8 attention heads. This matches the ML-20M benchmark configuration and item catalog from Guo et al. (2024) and Zhai et al. (2024).

**Results on ML-20M (50 epochs, dim=256, heads=8, batch=128, seed=42, full-catalog eval, one A10 GPU):**

| Blocks | Params | Test HR@10 | Test NDCG@10 | Time/epoch |
|--------|--------|------------|--------------|------------|
| 2 | 7.7M | 0.2938 | 0.1713 | ~73s |
| 4 | 8.5M | 0.3098 | 0.1830 | ~123s |
| 8 | 10.1M | 0.3196 | 0.1912 | ~222s |
| 16 | 13.2M | 0.3280 | 0.1970 | ~419s |

**Depth pays off.** Going from 2 to 16 blocks improves HR by +11.6% and NDCG by +15.0%. This is a clear, monotonic improvement across every depth we tested. Validation HR@10 was still rising at epoch 50 for every depth, so these runs are not fully converged.

**But depth has diminishing returns.** Each doubling of depth costs roughly 2x the training time but delivers similar or less improvement. At 16 blocks you're spending 5.7x the compute of 2 blocks for +12% quality. The natural next question is whether we can get more from the depth we already have rather than just stacking more blocks.

## v6: ReZero Residual Scaling

### Run

`uv run python train_v6.py --device cuda`

### Takeaway

**The problem with stacking blocks.** Each attention block transforms the input and adds its result back to a residual stream that flows through the whole model. With 2 blocks, this works fine. With 8, you have 8 randomly initialized transformations all adding noise to the stream before any of them have learned anything useful. The original item embeddings get buried. Training eventually sorts this out, but early epochs are wasted fighting through noise.

**ReZero's fix is almost embarrassingly simple.** Multiply each block's output by a learnable scalar, initialized to zero. At the start of training, every block is zeroed out, so the residual stream is just clean item embeddings flowing through. As training progresses, each scalar grows from zero, gradually turning on blocks as they learn something worth adding. The model decides for itself how much each block should contribute and when. Total cost: 16 extra parameters for 8 blocks.

**Results on ML-20M (dim=256, heads=8, 50 epochs, seed=42, full-catalog eval):**

| Config | Blocks | Params | Test HR@10 | Test NDCG@10 |
|--------|--------|--------|------------|--------------|
| v5 | 2 | 7.7M | 0.2938 | 0.1713 |
| v5 | 8 | 10.1M | 0.3196 | 0.1912 |
| v5 | 16 | 13.2M | 0.3280 | 0.1970 |
| v6 (ReZero) | 8 | 10.1M | 0.3287 | 0.1973 |

ReZero at 8 blocks matches plain SASRec at 16 blocks (0.3287 vs 0.3280) with 3.2M fewer parameters and just over half the training time per epoch (233s vs 419s). 16 scalar parameters buy as much as doubling the depth.

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

**Results on ML-20M (8 blocks, dim=256, heads=8, 50 epochs, full-catalog eval; v6 is seed 42, v7 is mean ± std over seeds 42-44):**

| Config | Test HR@10 | Test NDCG@10 |
|--------|------------|--------------|
| v6 (ReZero, absolute pos) | 0.3287 | 0.1973 |
| v7 (ReZero, relative + temporal) | 0.3380 ± 0.0009 | 0.2025 ± 0.0011 |

**Relative bias is as big a step as ReZero, and makes v7 the best model in the repo.** v7 beats v6 by +2.8% HR and +2.6% NDCG, the same +2.8% HR that ReZero added over plain 8-block SASRec. Unlike absolute position embeddings that encode "what slot is this," relative bias encodes "how far apart are these items" which is what actually matters for sequential patterns. The temporal bias adds real signal too, "items consumed in the same session" gets a different learned weight than "items months apart."

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

**Results on ML-20M (8 blocks, dim=256, heads=8, 50 epochs, full-catalog eval; seed 42 except v7, which is mean ± std over seeds 42-44):**

| Config | Test HR@10 | Test NDCG@10 |
|--------|------------|--------------|
| v5 blocks=8 (SASRec) | 0.3196 | 0.1912 |
| v6 (+ ReZero) | 0.3287 | 0.1973 |
| **v7 (+ relative bias)** | **0.3380 ± 0.0009** | **0.2025 ± 0.0011** |
| v8 (HSTU) | 0.3236 | 0.1929 |
| SASRec (Zhai et al., 2024) | 0.2889 | 0.1621 |
| HSTU (Zhai et al., 2024) | 0.3273 | 0.1895 |
| HSTU-large (Zhai et al., 2024) | 0.3556 | 0.2098 |
| HSTU, recomputed (Tikhonovich et al., 2025) | 0.3464 | 0.2040 |
| eSASRec, 8 blocks (Tikhonovich et al., 2025) | 0.3457 | 0.2107 |

**The architecture swap is not a clear win at this scale.** v8 HSTU (0.3236) beats the v5 8-block SASRec baseline (0.3196), but lands below v6 ReZero (0.3287) and v7 relative bias (0.3380). The HSTU block replaces softmax+FFN with SiLU+gating while keeping the same relative bias infrastructure from v7.

**HSTU starts fast, then flattens.** The validation curves show a pattern the final numbers hide. At epoch 10, v8 is already ahead of v7 (validation HR@10 0.3291 against 0.3281), and v9 leads every model in the repo (0.3356). By epoch 50 the order has reversed. Over their last 10 epochs, v8 and v10 gained only about 0.2 points of validation HR@10, while v5 (8 and 16 blocks), v6 and v7 gained 0.34 to 0.58 points and were still climbing. v9 sits in between: 0.16 to 0.64 points depending on the seed.

| Validation HR@10 at epoch (seed 42) | 10 | 20 | 30 | 40 | 50 |
|-------------------------------------|----|----|----|----|----|
| v5 blocks=8 (SASRec) | 0.2974 | 0.3242 | 0.3347 | 0.3447 | 0.3504 |
| v7 (+ relative bias) | 0.3281 | 0.3506 | 0.3606 | 0.3684 | 0.3729 |
| v8 (HSTU) | 0.3291 | 0.3409 | 0.3495 | 0.3561 | 0.3581 |
| v9 (+ φ1) | 0.3356 | 0.3510 | 0.3608 | 0.3674 | 0.3690 |

So 50 epochs undertrains the SASRec line more than the HSTU line. We checked by re-running v5 (8 blocks), v7 and v8 for 100 epochs (seed 42). All three gained 0.8 to 1.0 points of test HR@10, and v7's lead over v8 grew from 1.4 to 1.7 points:

| Test at 100 epochs (seed 42) | HR@10 | NDCG@10 | HR@10 at 50 epochs |
|------------------------------|-------|---------|--------------------|
| v5 blocks=8 (SASRec) | 0.3297 | 0.1976 | 0.3196 |
| v7 (+ relative bias) | 0.3483 | 0.2102 | 0.3381 |
| v8 (HSTU) | 0.3313 | 0.1976 | 0.3236 |

v8 still learns faster early on, but v5 catches up: v8's lead over the same-depth SASRec shrinks from 0.4 points to under 0.2.

**How this compares with published numbers.** Our catalog now matches Zhai et al. (2024), but the tie order doesn't. Their preprocessing reorders interactions with tied timestamps, which changes about 25% of the test set (Tikhonovich et al., 2025). eSASRec re-ran HSTU's original code on the order-preserving split that `data.py` produces, so the recomputed rows are the fair reference. Against them, after 50 epochs, v7 is 0.8 points of HR@10 behind both eSASRec and the recomputed HSTU, while our v8 HSTU is 2.3 points behind the recomputed HSTU. eSASRec trains for up to 100 epochs, so the fair budget is 100, not 50. At 100 epochs v7 reaches 0.3483, above both eSASRec (0.3457) and the recomputed HSTU (0.3464). Giving v7 eSASRec's widths (4× feed-forward, 16 heads, 256 negatives) does not help: 0.3476, with 30% more parameters and 1.8× the time per epoch. More epochs do not rescue our HSTU, though: v8 at 100 epochs (0.3313) is still 1.5 points behind the recomputed HSTU. One known difference is long-sequence handling, which the reference training uses and nanoSASRec does not yet have (see Stochastic Length training in the README roadmap). A well-trained SASRec with ReZero and relative bias beats our own HSTU implementation and, given the same training budget, matches the best published results on this split (eSASRec and the recomputed HSTU) without changing the attention mechanism. HSTU-large (0.3556) is higher, but it is a bigger model and was evaluated on Zhai et al.'s reordered split.

## v9: φ1 SwishLayerNorm Pre-Activation

### Run

`uv run python train_v9.py --device cuda`

### Takeaway

**What changed.** Combined the four separate Q, K, V, U linear projections into a single projection (dim to 4*dim), then applied SwishLayerNorm before splitting them apart. SwishLayerNorm is `x * sigmoid(LayerNorm(x))`: LayerNorm normalizes the combined projection, sigmoid gates each dimension between 0 and 1, and multiplying by the original lets the model decide how much to keep.

This matters because the projections must be combined. LayerNorm computes statistics (mean, variance) across the full 4*dim dimension, so Q's normalization depends on what U, V, and K look like. Separate projections with separate LayerNorms would be a different operation.

The result is that Q, K, V, U become nonlinear functions of the input before they interact. In v8, attention scores were a bilinear form (linear Q dotted with linear K). In v9, both Q and K pass through a nonlinearity first, giving the model a richer set of patterns it can represent.

**Results on ML-20M (8 blocks, dim=256, heads=8, 50 epochs, full-catalog eval; v8 is seed 42, v9 is mean ± std over seeds 42-44):**

| Config | Params | Test HR@10 | Test NDCG@10 |
|--------|--------|------------|--------------|
| v8 (HSTU, separate projections) | 9.5M | 0.3236 | 0.1929 |
| **v9 (HSTU, combined + φ1)** | **9.5M** | **0.3354 ± 0.0009** | **0.2008 ± <0.0001** |
| HSTU (Zhai et al., 2024) | — | 0.3273 | 0.1895 |
| HSTU, recomputed (Tikhonovich et al., 2025) | — | 0.3464 | 0.2040 |

**φ1 improved over v8 by +3.6% HR and +4.1% NDCG.** That recovers 82% of v8's gap to v7 and brings v9 within 1.1 points of HR@10 of the recomputed HSTU. v9 also learns fastest: at epoch 10 its validation HR@10 is 0.3356, against 0.3281 for v7, though v7 draws level by epoch 30 and passes it by epoch 40. But v7 SASRec+ReZero+relative bias (0.3380) still leads the repo, reinforcing that training and positional improvements matter more than the attention mechanism at this scale.

## v10: Training Alignment

### Run

`uv run python train_v10.py --device cuda`

### Takeaway

**What changed.** Three training details from the Zhai et al. (2024) HSTU setup, applied on top of v9's architecture:

**L2 normalization.** Normalize both the model's output embeddings and the item embeddings to unit length before scoring. This turns dot products into cosine similarity, bounded between [-1, 1]. Without it, the model can give popular items larger embedding norms so they score high regardless of relevance. With it, all items compete on direction (angular similarity) alone.

**Temperature scaling.** Divide all logits by a temperature parameter (0.05) before the softmax loss. With cosine similarity in [-1, 1], dividing by 0.05 scales logits to [-20, 20]. This forces the softmax to commit hard to its top picks. A higher temperature (softer distribution) lets the model be vague about rankings.

**Collision masking.** If a randomly sampled negative happens to be the same item as the positive target, mask it to negative infinity before the loss. Without this, the model occasionally gets penalized for scoring a correct item highly. With 512 negatives out of 26,744 items, the positive lands among the negatives about 1.9% of the time per position.

These are standard techniques in production recommendation systems and appear in the Zhai et al. (2024) HSTU training setup.

**Results on ML-20M (8 blocks, dim=256, heads=8, 50 epochs, full-catalog eval; v10 is seed 42, v9 is mean ± std over seeds 42-44):**

| Config | HR@10 | NDCG@10 |
|--------|-------|---------|
| v9 (raw dot products) | 0.3354 ± 0.0009 | 0.2008 ± <0.0001 |
| v10 temp=0.05 | 0.3269 | 0.1949 |

**These techniques didn't help at our scale.** v10 underperformed v9 by 0.9 points of HR@10, as it did on the v0.2.1 catalog across every temperature tried there. The likely reason is the negative sampling strategy: we use random negatives, while Zhai et al. (2024) use in-batch negatives (the positive items from other users in the same batch).

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

**Results on ML-20M (8 blocks, dim=256, heads=8, 50 epochs, full-catalog eval; v11 is seed 42, v9 is mean ± std over seeds 42-44):**

| Variant | HR@10 | NDCG@10 |
|---------|-------|---------|
| v9 (next-item only) | 0.3354 ± 0.0009 | 0.2008 ± <0.0001 |
| Dense All Action (K=10 horizon) | 0.2507 | 0.1375 |

**Why it doesn't work for next-item evaluation.** For a sequence of length 199 with horizon K=10, each position predicts its next 10 items. The next-item prediction (the diagonal) is 1 out of 10 targets, so only 10% of the gradient signal targets the metric we actually evaluate on. The other 90% optimizes for predicting items 2-10 steps ahead.

**What Dense All Action is actually for.** PinnerFormer introduced this technique for temporal embedding stability at Pinterest. If the model can predict not just the next item but all future items, then user embeddings computed today are still useful for serving recommendations tomorrow. The evaluation in PinnerFormer reflects this: they measure whether embeddings from 24 hours ago still produce good recommendations today. That's a different objective than next-item HR@10.

**The open-source generative-recommenders implementation does not use Dense All Action for the ML-20M benchmark.** The Zhai et al. (2024) ML-20M results (HR@10 0.3273) use standard next-item sampled softmax loss, the same as our v9. Dense All Action is referenced conceptually but is not part of the benchmark code.
