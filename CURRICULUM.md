# nanoSASRec Walkthrough

Curriculum walkthrough from v1 to v8.

## Differences

Here are known differences than the original implementation of SASRec.

- **v1 uses Pre-LN** (norm before attention), the original uses Post-LN (norm after residual add). Pre-LN is more stable for deeper models, which is why we start with it — v5+ needs it to scale.
- **Weight initialization** differs from the original (PyTorch defaults vs Xavier uniform). Affects early training dynamics, shifts final metrics by 1-3%.

These are intentional choices, not bugs.

## v1

### Run

`uv run python train_v1.py --device mps --epochs 200 --eval-every 20`

### Output

```bash
Number Users: 6,040
Number Items: 3,416
Dataset Lengths: 6040 6040 6040
Model parameters: 211,500
Est. batches per epoch: 47
[20:19:51] [Epoch 1/200] Avg BCE loss: 5.3195
[20:19:53] [Epoch 2/200] Avg BCE loss: 3.8279
[20:19:55] [Epoch 3/200] Avg BCE loss: 2.7345
[20:19:57] [Epoch 4/200] Avg BCE loss: 2.0885
[20:19:59] [Epoch 5/200] Avg BCE loss: 1.7289
[20:20:00] [Epoch 6/200] Avg BCE loss: 1.5222
[20:20:02] [Epoch 7/200] Avg BCE loss: 1.4010
[20:20:04] [Epoch 8/200] Avg BCE loss: 1.3197
[20:20:06] [Epoch 9/200] Avg BCE loss: 1.2715
[20:20:08] [Epoch 10/200] Avg BCE loss: 1.2333
[20:20:10] [Epoch 11/200] Avg BCE loss: 1.2062
[20:20:12] [Epoch 12/200] Avg BCE loss: 1.1757
[20:20:14] [Epoch 13/200] Avg BCE loss: 1.1601
[20:20:16] [Epoch 14/200] Avg BCE loss: 1.1430
[20:20:18] [Epoch 15/200] Avg BCE loss: 1.1258
[20:20:20] [Epoch 16/200] Avg BCE loss: 1.1049
[20:20:22] [Epoch 17/200] Avg BCE loss: 1.0888
[20:20:23] [Epoch 18/200] Avg BCE loss: 1.0829
[20:20:25] [Epoch 19/200] Avg BCE loss: 1.0714
[20:20:36] [Epoch 20/200] Avg BCE loss: 1.0653 | HR@10: 0.4379 NDCG@10: 0.2416
[20:20:38] [Epoch 21/200] Avg BCE loss: 1.0551
[20:20:40] [Epoch 22/200] Avg BCE loss: 1.0517
[20:20:41] [Epoch 23/200] Avg BCE loss: 1.0451
[20:20:43] [Epoch 24/200] Avg BCE loss: 1.0407
[20:20:45] [Epoch 25/200] Avg BCE loss: 1.0380
[20:20:47] [Epoch 26/200] Avg BCE loss: 1.0310
[20:20:49] [Epoch 27/200] Avg BCE loss: 1.0285
[20:20:51] [Epoch 28/200] Avg BCE loss: 1.0232
[20:20:52] [Epoch 29/200] Avg BCE loss: 1.0174
[20:20:54] [Epoch 30/200] Avg BCE loss: 1.0169
[20:20:56] [Epoch 31/200] Avg BCE loss: 1.0125
[20:20:58] [Epoch 32/200] Avg BCE loss: 1.0126
[20:21:00] [Epoch 33/200] Avg BCE loss: 1.0077
[20:21:01] [Epoch 34/200] Avg BCE loss: 1.0095
[20:21:03] [Epoch 35/200] Avg BCE loss: 1.0086
[20:21:05] [Epoch 36/200] Avg BCE loss: 1.0048
[20:21:07] [Epoch 37/200] Avg BCE loss: 1.0016
[20:21:09] [Epoch 38/200] Avg BCE loss: 1.0047
[20:21:11] [Epoch 39/200] Avg BCE loss: 0.9993
[20:21:21] [Epoch 40/200] Avg BCE loss: 0.9934 | HR@10: 0.4429 NDCG@10: 0.2423
[20:21:23] [Epoch 41/200] Avg BCE loss: 0.9968
[20:21:24] [Epoch 42/200] Avg BCE loss: 0.9947
[20:21:26] [Epoch 43/200] Avg BCE loss: 0.9943
[20:21:28] [Epoch 44/200] Avg BCE loss: 0.9914
[20:21:30] [Epoch 45/200] Avg BCE loss: 0.9887
[20:21:32] [Epoch 46/200] Avg BCE loss: 0.9835
[20:21:33] [Epoch 47/200] Avg BCE loss: 0.9760
[20:21:35] [Epoch 48/200] Avg BCE loss: 0.9761
[20:21:37] [Epoch 49/200] Avg BCE loss: 0.9717
[20:21:39] [Epoch 50/200] Avg BCE loss: 0.9652
[20:21:41] [Epoch 51/200] Avg BCE loss: 0.9565
[20:21:42] [Epoch 52/200] Avg BCE loss: 0.9568
[20:21:44] [Epoch 53/200] Avg BCE loss: 0.9499
[20:21:46] [Epoch 54/200] Avg BCE loss: 0.9459
[20:21:48] [Epoch 55/200] Avg BCE loss: 0.9357
[20:21:50] [Epoch 56/200] Avg BCE loss: 0.9276
[20:21:52] [Epoch 57/200] Avg BCE loss: 0.9204
[20:21:53] [Epoch 58/200] Avg BCE loss: 0.9136
[20:21:55] [Epoch 59/200] Avg BCE loss: 0.9086
[20:22:05] [Epoch 60/200] Avg BCE loss: 0.8979 | HR@10: 0.5432 NDCG@10: 0.3061
[20:22:07] [Epoch 61/200] Avg BCE loss: 0.8939
[20:22:09] [Epoch 62/200] Avg BCE loss: 0.8850
[20:22:11] [Epoch 63/200] Avg BCE loss: 0.8796
[20:22:13] [Epoch 64/200] Avg BCE loss: 0.8702
[20:22:14] [Epoch 65/200] Avg BCE loss: 0.8665
[20:22:16] [Epoch 66/200] Avg BCE loss: 0.8631
[20:22:18] [Epoch 67/200] Avg BCE loss: 0.8571
[20:22:20] [Epoch 68/200] Avg BCE loss: 0.8420
[20:22:22] [Epoch 69/200] Avg BCE loss: 0.8430
[20:22:23] [Epoch 70/200] Avg BCE loss: 0.8383
[20:22:25] [Epoch 71/200] Avg BCE loss: 0.8315
[20:22:27] [Epoch 72/200] Avg BCE loss: 0.8244
[20:22:29] [Epoch 73/200] Avg BCE loss: 0.8264
[20:22:31] [Epoch 74/200] Avg BCE loss: 0.8210
[20:22:33] [Epoch 75/200] Avg BCE loss: 0.8145
[20:22:34] [Epoch 76/200] Avg BCE loss: 0.8117
[20:22:36] [Epoch 77/200] Avg BCE loss: 0.8042
[20:22:38] [Epoch 78/200] Avg BCE loss: 0.7982
[20:22:40] [Epoch 79/200] Avg BCE loss: 0.7992
[20:22:50] [Epoch 80/200] Avg BCE loss: 0.7939 | HR@10: 0.6382 NDCG@10: 0.3778
[20:22:52] [Epoch 81/200] Avg BCE loss: 0.7884
[20:22:54] [Epoch 82/200] Avg BCE loss: 0.7877
[20:22:55] [Epoch 83/200] Avg BCE loss: 0.7825
[20:22:57] [Epoch 84/200] Avg BCE loss: 0.7812
[20:22:59] [Epoch 85/200] Avg BCE loss: 0.7758
[20:23:01] [Epoch 86/200] Avg BCE loss: 0.7734
[20:23:03] [Epoch 87/200] Avg BCE loss: 0.7634
[20:23:05] [Epoch 88/200] Avg BCE loss: 0.7598
[20:23:06] [Epoch 89/200] Avg BCE loss: 0.7620
[20:23:08] [Epoch 90/200] Avg BCE loss: 0.7595
[20:23:10] [Epoch 91/200] Avg BCE loss: 0.7506
[20:23:12] [Epoch 92/200] Avg BCE loss: 0.7501
[20:23:14] [Epoch 93/200] Avg BCE loss: 0.7474
[20:23:16] [Epoch 94/200] Avg BCE loss: 0.7422
[20:23:18] [Epoch 95/200] Avg BCE loss: 0.7430
[20:23:19] [Epoch 96/200] Avg BCE loss: 0.7396
[20:23:21] [Epoch 97/200] Avg BCE loss: 0.7390
[20:23:23] [Epoch 98/200] Avg BCE loss: 0.7337
[20:23:25] [Epoch 99/200] Avg BCE loss: 0.7342
[20:23:35] [Epoch 100/200] Avg BCE loss: 0.7321 | HR@10: 0.6932 NDCG@10: 0.4150
[20:23:37] [Epoch 101/200] Avg BCE loss: 0.7268
[20:23:39] [Epoch 102/200] Avg BCE loss: 0.7247
[20:23:41] [Epoch 103/200] Avg BCE loss: 0.7198
[20:23:43] [Epoch 104/200] Avg BCE loss: 0.7238
[20:23:44] [Epoch 105/200] Avg BCE loss: 0.7218
[20:23:46] [Epoch 106/200] Avg BCE loss: 0.7176
[20:23:48] [Epoch 107/200] Avg BCE loss: 0.7140
[20:23:50] [Epoch 108/200] Avg BCE loss: 0.7154
[20:23:52] [Epoch 109/200] Avg BCE loss: 0.7148
[20:23:54] [Epoch 110/200] Avg BCE loss: 0.7095
[20:23:56] [Epoch 111/200] Avg BCE loss: 0.7102
[20:23:57] [Epoch 112/200] Avg BCE loss: 0.7033
[20:23:59] [Epoch 113/200] Avg BCE loss: 0.7041
[20:24:01] [Epoch 114/200] Avg BCE loss: 0.7067
[20:24:03] [Epoch 115/200] Avg BCE loss: 0.6991
[20:24:05] [Epoch 116/200] Avg BCE loss: 0.7038
[20:24:07] [Epoch 117/200] Avg BCE loss: 0.6941
[20:24:09] [Epoch 118/200] Avg BCE loss: 0.6946
[20:24:11] [Epoch 119/200] Avg BCE loss: 0.6941
[20:24:21] [Epoch 120/200] Avg BCE loss: 0.6895 | HR@10: 0.7119 NDCG@10: 0.4384
[20:24:23] [Epoch 121/200] Avg BCE loss: 0.6888
[20:24:25] [Epoch 122/200] Avg BCE loss: 0.6858
[20:24:27] [Epoch 123/200] Avg BCE loss: 0.6866
[20:24:29] [Epoch 124/200] Avg BCE loss: 0.6888
[20:24:31] [Epoch 125/200] Avg BCE loss: 0.6853
[20:24:33] [Epoch 126/200] Avg BCE loss: 0.6830
[20:24:34] [Epoch 127/200] Avg BCE loss: 0.6788
[20:24:36] [Epoch 128/200] Avg BCE loss: 0.6805
[20:24:38] [Epoch 129/200] Avg BCE loss: 0.6767
[20:24:40] [Epoch 130/200] Avg BCE loss: 0.6732
[20:24:42] [Epoch 131/200] Avg BCE loss: 0.6672
[20:24:44] [Epoch 132/200] Avg BCE loss: 0.6694
[20:24:45] [Epoch 133/200] Avg BCE loss: 0.6707
[20:24:47] [Epoch 134/200] Avg BCE loss: 0.6692
[20:24:49] [Epoch 135/200] Avg BCE loss: 0.6664
[20:24:51] [Epoch 136/200] Avg BCE loss: 0.6675
[20:24:53] [Epoch 137/200] Avg BCE loss: 0.6590
[20:24:54] [Epoch 138/200] Avg BCE loss: 0.6631
[20:24:56] [Epoch 139/200] Avg BCE loss: 0.6621
[20:25:07] [Epoch 140/200] Avg BCE loss: 0.6559 | HR@10: 0.7387 NDCG@10: 0.4662
[20:25:09] [Epoch 141/200] Avg BCE loss: 0.6530
[20:25:11] [Epoch 142/200] Avg BCE loss: 0.6559
[20:25:12] [Epoch 143/200] Avg BCE loss: 0.6529
[20:25:14] [Epoch 144/200] Avg BCE loss: 0.6479
[20:25:16] [Epoch 145/200] Avg BCE loss: 0.6526
[20:25:18] [Epoch 146/200] Avg BCE loss: 0.6485
[20:25:20] [Epoch 147/200] Avg BCE loss: 0.6474
[20:25:22] [Epoch 148/200] Avg BCE loss: 0.6497
[20:25:23] [Epoch 149/200] Avg BCE loss: 0.6435
[20:25:25] [Epoch 150/200] Avg BCE loss: 0.6472
[20:25:27] [Epoch 151/200] Avg BCE loss: 0.6422
[20:25:29] [Epoch 152/200] Avg BCE loss: 0.6384
[20:25:31] [Epoch 153/200] Avg BCE loss: 0.6427
[20:25:33] [Epoch 154/200] Avg BCE loss: 0.6421
[20:25:35] [Epoch 155/200] Avg BCE loss: 0.6385
[20:25:36] [Epoch 156/200] Avg BCE loss: 0.6384
[20:25:38] [Epoch 157/200] Avg BCE loss: 0.6366
[20:25:40] [Epoch 158/200] Avg BCE loss: 0.6363
[20:25:42] [Epoch 159/200] Avg BCE loss: 0.6361
[20:25:53] [Epoch 160/200] Avg BCE loss: 0.6330 | HR@10: 0.7505 NDCG@10: 0.4781
[20:25:55] [Epoch 161/200] Avg BCE loss: 0.6236
[20:25:56] [Epoch 162/200] Avg BCE loss: 0.6326
[20:25:58] [Epoch 163/200] Avg BCE loss: 0.6239
[20:26:00] [Epoch 164/200] Avg BCE loss: 0.6294
[20:26:02] [Epoch 165/200] Avg BCE loss: 0.6264
[20:26:04] [Epoch 166/200] Avg BCE loss: 0.6248
[20:26:06] [Epoch 167/200] Avg BCE loss: 0.6273
[20:26:07] [Epoch 168/200] Avg BCE loss: 0.6220
[20:26:09] [Epoch 169/200] Avg BCE loss: 0.6192
[20:26:11] [Epoch 170/200] Avg BCE loss: 0.6140
[20:26:13] [Epoch 171/200] Avg BCE loss: 0.6221
[20:26:15] [Epoch 172/200] Avg BCE loss: 0.6178
[20:26:17] [Epoch 173/200] Avg BCE loss: 0.6211
[20:26:18] [Epoch 174/200] Avg BCE loss: 0.6191
[20:26:20] [Epoch 175/200] Avg BCE loss: 0.6152
[20:26:22] [Epoch 176/200] Avg BCE loss: 0.6140
[20:26:24] [Epoch 177/200] Avg BCE loss: 0.6108
[20:26:26] [Epoch 178/200] Avg BCE loss: 0.6122
[20:26:27] [Epoch 179/200] Avg BCE loss: 0.6161
[20:26:38] [Epoch 180/200] Avg BCE loss: 0.6144 | HR@10: 0.7649 NDCG@10: 0.4985
[20:26:40] [Epoch 181/200] Avg BCE loss: 0.6152
[20:26:42] [Epoch 182/200] Avg BCE loss: 0.6081
[20:26:43] [Epoch 183/200] Avg BCE loss: 0.6125
[20:26:45] [Epoch 184/200] Avg BCE loss: 0.6066
[20:26:47] [Epoch 185/200] Avg BCE loss: 0.6090
[20:26:49] [Epoch 186/200] Avg BCE loss: 0.6027
[20:26:51] [Epoch 187/200] Avg BCE loss: 0.6053
[20:26:53] [Epoch 188/200] Avg BCE loss: 0.5990
[20:26:54] [Epoch 189/200] Avg BCE loss: 0.6057
[20:26:56] [Epoch 190/200] Avg BCE loss: 0.6027
[20:26:58] [Epoch 191/200] Avg BCE loss: 0.5993
[20:27:00] [Epoch 192/200] Avg BCE loss: 0.6005
[20:27:02] [Epoch 193/200] Avg BCE loss: 0.6045
[20:27:03] [Epoch 194/200] Avg BCE loss: 0.5941
[20:27:05] [Epoch 195/200] Avg BCE loss: 0.6002
[20:27:07] [Epoch 196/200] Avg BCE loss: 0.5951
[20:27:09] [Epoch 197/200] Avg BCE loss: 0.5972
[20:27:11] [Epoch 198/200] Avg BCE loss: 0.5919
[20:27:13] [Epoch 199/200] Avg BCE loss: 0.5967
[20:27:23] [Epoch 200/200] Avg BCE loss: 0.5940 | HR@10: 0.7705 NDCG@10: 0.5112
[20:27:32] [Epochs 200] Test metrics: HR@10: 0.7490 NDCG@10: 0.4823

  Prediction demo (user 1):
    Recent history:  Toy Story (1995), Bug's Life, A (1998), Antz (1998), Hunchback of Notre Dame, The (1996), Hercules (1997)
    Ground truth:    Pocahontas (1995)
    Model top-10:
       1. Lion King, The (1994)
       2. South Park: Bigger, Longer and Uncut (1999)
       3. Babe (1995)
       4. Nightmare Before Christmas, The (1993)
       5. Grosse Pointe Blank (1997)
       6. Clerks (1994)
       7. Groundhog Day (1993)
       8. Austin Powers: International Man of Mystery (1997)
       9. Fantasia (1940)
      10. Mask, The (1994)
```

### Takeaway 

**What we built:** A causal Transformer decoder (SASRec) trained with binary cross-entropy loss and 1 negative per positive, evaluated with leave-one-out ranking against 100 random negatives.

**What we learned about the architecture:** Self-attention lets each position in a user's history decide which past items are most relevant for predicting what comes next. The shifted-sequence trick gives us one training signal per position, not just one per user. Residual connections and layer normalization keep gradients flowing through stacked blocks.

**What the metrics mean:** HR@10 = 0.77 means for 77% of users, the model ranked the actual next item within the top 10 out of 101 candidates. NDCG@10 = 0.51 means that when it did rank correctly, the item tended to land around position 2-3 rather than position 10.

**What's wrong — the overconfidence problem:** The loss dropped to 0.59, which sounds like "almost solved." But map that through the BCE loss curve: a loss of 0.59 means the model assigns P ≈ 0.95 to positive items on average. By the end of training, the top 25 items for any user all score above 0.99. When everything is "99% likely," the model can't distinguish between rank 1 and rank 25 — they all look the same. This is overconfidence.

**What 0.99 actually means:** The model is saying "I am 99% sure this is the right next item" for dozens of items simultaneously. That's mathematically incoherent — they can't all be 99% likely. In a catalog of 3,416 items, the true probability of any single item being next is roughly 1/3,416 ≈ 0.03%. A well-calibrated model should predict something close to that for most items and maybe 5-20% for the top few. Predicting 0.99 is overconfident by a factor of ~3,000×.

**Why this happens with BCE:** BCE treats each item as an independent binary question — "is this item next? yes/no." With only 1 negative, the model easily distinguishes positive from negative and pushes its confidence to extremes. It never has to answer the harder question: "which of these items is most likely next?" That competition between candidates is what v2 introduces.

---

## v2

### Run

`uv run python train_v2.py --device mps --epochs 200 --eval-every 20`

### Output

```bash
Number Users: 6,040
Number Items: 3,416
Dataset Lengths: 6040 6040 6040
Model parameters: 211,500
Estimate batches per epoch: 47
[20:52:19] [Epoch 1/200] Avg loss: 17.9515
[20:52:21] [Epoch 2/200] Avg loss: 12.2586
[20:52:22] [Epoch 3/200] Avg loss: 9.2592
[20:52:24] [Epoch 4/200] Avg loss: 7.5734
[20:52:25] [Epoch 5/200] Avg loss: 6.5830
[20:52:26] [Epoch 6/200] Avg loss: 6.0471
[20:52:28] [Epoch 7/200] Avg loss: 5.7060
[20:52:30] [Epoch 8/200] Avg loss: 5.5073
[20:52:31] [Epoch 9/200] Avg loss: 5.3852
[20:52:33] [Epoch 10/200] Avg loss: 5.2727
[20:52:34] [Epoch 11/200] Avg loss: 5.1895
[20:52:36] [Epoch 12/200] Avg loss: 5.0987
[20:52:37] [Epoch 13/200] Avg loss: 5.0235
[20:52:39] [Epoch 14/200] Avg loss: 4.9657
[20:52:40] [Epoch 15/200] Avg loss: 4.9136
[20:52:42] [Epoch 16/200] Avg loss: 4.8784
[20:52:43] [Epoch 17/200] Avg loss: 4.8390
[20:52:44] [Epoch 18/200] Avg loss: 4.8038
[20:52:46] [Epoch 19/200] Avg loss: 4.7765
[20:52:56] [Epoch 20/200] Avg loss: 4.7642 | HR@10: 0.4373 NDCG@10: 0.2416
[20:52:57] [Epoch 21/200] Avg loss: 4.7382
[20:52:59] [Epoch 22/200] Avg loss: 4.7354
[20:53:00] [Epoch 23/200] Avg loss: 4.7201
[20:53:02] [Epoch 24/200] Avg loss: 4.7257
[20:53:03] [Epoch 25/200] Avg loss: 4.7049
[20:53:05] [Epoch 26/200] Avg loss: 4.6967
[20:53:06] [Epoch 27/200] Avg loss: 4.6946
[20:53:08] [Epoch 28/200] Avg loss: 4.6807
[20:53:09] [Epoch 29/200] Avg loss: 4.6834
[20:53:11] [Epoch 30/200] Avg loss: 4.6830
[20:53:12] [Epoch 31/200] Avg loss: 4.6676
[20:53:14] [Epoch 32/200] Avg loss: 4.6553
[20:53:15] [Epoch 33/200] Avg loss: 4.6473
[20:53:17] [Epoch 34/200] Avg loss: 4.6308
[20:53:18] [Epoch 35/200] Avg loss: 4.6170
[20:53:20] [Epoch 36/200] Avg loss: 4.6047
[20:53:21] [Epoch 37/200] Avg loss: 4.5979
[20:53:23] [Epoch 38/200] Avg loss: 4.5617
[20:53:24] [Epoch 39/200] Avg loss: 4.5541
[20:53:35] [Epoch 40/200] Avg loss: 4.5315 | HR@10: 0.4735 NDCG@10: 0.2639
[20:53:36] [Epoch 41/200] Avg loss: 4.5036
[20:53:38] [Epoch 42/200] Avg loss: 4.4807
[20:53:39] [Epoch 43/200] Avg loss: 4.4614
[20:53:41] [Epoch 44/200] Avg loss: 4.4179
[20:53:42] [Epoch 45/200] Avg loss: 4.4048
[20:53:44] [Epoch 46/200] Avg loss: 4.3812
[20:53:45] [Epoch 47/200] Avg loss: 4.3600
[20:53:47] [Epoch 48/200] Avg loss: 4.3362
[20:53:48] [Epoch 49/200] Avg loss: 4.3038
[20:53:50] [Epoch 50/200] Avg loss: 4.2862
[20:53:51] [Epoch 51/200] Avg loss: 4.2582
[20:53:53] [Epoch 52/200] Avg loss: 4.2470
[20:53:54] [Epoch 53/200] Avg loss: 4.2352
[20:53:56] [Epoch 54/200] Avg loss: 4.2123
[20:53:57] [Epoch 55/200] Avg loss: 4.1882
[20:53:59] [Epoch 56/200] Avg loss: 4.1648
[20:54:00] [Epoch 57/200] Avg loss: 4.1489
[20:54:02] [Epoch 58/200] Avg loss: 4.1299
[20:54:03] [Epoch 59/200] Avg loss: 4.1213
[20:54:14] [Epoch 60/200] Avg loss: 4.0941 | HR@10: 0.6174 NDCG@10: 0.3705
[20:54:16] [Epoch 61/200] Avg loss: 4.0713
[20:54:17] [Epoch 62/200] Avg loss: 4.0643
[20:54:19] [Epoch 63/200] Avg loss: 4.0370
[20:54:20] [Epoch 64/200] Avg loss: 4.0245
[20:54:22] [Epoch 65/200] Avg loss: 4.0153
[20:54:23] [Epoch 66/200] Avg loss: 3.9992
[20:54:25] [Epoch 67/200] Avg loss: 3.9829
[20:54:26] [Epoch 68/200] Avg loss: 3.9508
[20:54:28] [Epoch 69/200] Avg loss: 3.9368
[20:54:30] [Epoch 70/200] Avg loss: 3.9325
[20:54:31] [Epoch 71/200] Avg loss: 3.9151
[20:54:33] [Epoch 72/200] Avg loss: 3.8937
[20:54:34] [Epoch 73/200] Avg loss: 3.8730
[20:54:36] [Epoch 74/200] Avg loss: 3.8584
[20:54:37] [Epoch 75/200] Avg loss: 3.8291
[20:54:39] [Epoch 76/200] Avg loss: 3.8325
[20:54:40] [Epoch 77/200] Avg loss: 3.8186
[20:54:42] [Epoch 78/200] Avg loss: 3.8064
[20:54:43] [Epoch 79/200] Avg loss: 3.7860
[20:54:54] [Epoch 80/200] Avg loss: 3.7955 | HR@10: 0.6823 NDCG@10: 0.4325
[20:54:55] [Epoch 81/200] Avg loss: 3.7619
[20:54:57] [Epoch 82/200] Avg loss: 3.7368
[20:54:58] [Epoch 83/200] Avg loss: 3.7112
[20:55:00] [Epoch 84/200] Avg loss: 3.7089
[20:55:01] [Epoch 85/200] Avg loss: 3.7043
[20:55:03] [Epoch 86/200] Avg loss: 3.7033
[20:55:04] [Epoch 87/200] Avg loss: 3.6646
[20:55:06] [Epoch 88/200] Avg loss: 3.6416
[20:55:07] [Epoch 89/200] Avg loss: 3.6618
[20:55:09] [Epoch 90/200] Avg loss: 3.6341
[20:55:10] [Epoch 91/200] Avg loss: 3.6497
[20:55:12] [Epoch 92/200] Avg loss: 3.6035
[20:55:13] [Epoch 93/200] Avg loss: 3.6099
[20:55:15] [Epoch 94/200] Avg loss: 3.5969
[20:55:16] [Epoch 95/200] Avg loss: 3.5824
[20:55:18] [Epoch 96/200] Avg loss: 3.5791
[20:55:19] [Epoch 97/200] Avg loss: 3.5567
[20:55:21] [Epoch 98/200] Avg loss: 3.5575
[20:55:22] [Epoch 99/200] Avg loss: 3.5649
[20:55:33] [Epoch 100/200] Avg loss: 3.5460 | HR@10: 0.7298 NDCG@10: 0.4823
[20:55:34] [Epoch 101/200] Avg loss: 3.5260
[20:55:36] [Epoch 102/200] Avg loss: 3.5138
[20:55:37] [Epoch 103/200] Avg loss: 3.5150
[20:55:39] [Epoch 104/200] Avg loss: 3.5134
[20:55:40] [Epoch 105/200] Avg loss: 3.5115
[20:55:42] [Epoch 106/200] Avg loss: 3.4852
[20:55:43] [Epoch 107/200] Avg loss: 3.4672
[20:55:45] [Epoch 108/200] Avg loss: 3.4568
[20:55:46] [Epoch 109/200] Avg loss: 3.4598
[20:55:48] [Epoch 110/200] Avg loss: 3.4523
[20:55:49] [Epoch 111/200] Avg loss: 3.4468
[20:55:51] [Epoch 112/200] Avg loss: 3.4271
[20:55:52] [Epoch 113/200] Avg loss: 3.4247
[20:55:54] [Epoch 114/200] Avg loss: 3.4201
[20:55:55] [Epoch 115/200] Avg loss: 3.4007
[20:55:57] [Epoch 116/200] Avg loss: 3.3905
[20:55:58] [Epoch 117/200] Avg loss: 3.3836
[20:56:00] [Epoch 118/200] Avg loss: 3.3914
[20:56:01] [Epoch 119/200] Avg loss: 3.3849
[20:56:11] [Epoch 120/200] Avg loss: 3.3573 | HR@10: 0.7533 NDCG@10: 0.5106
[20:56:13] [Epoch 121/200] Avg loss: 3.3767
[20:56:14] [Epoch 122/200] Avg loss: 3.3605
[20:56:16] [Epoch 123/200] Avg loss: 3.3540
[20:56:17] [Epoch 124/200] Avg loss: 3.3525
[20:56:19] [Epoch 125/200] Avg loss: 3.3347
[20:56:20] [Epoch 126/200] Avg loss: 3.3495
[20:56:22] [Epoch 127/200] Avg loss: 3.3513
[20:56:24] [Epoch 128/200] Avg loss: 3.3333
[20:56:25] [Epoch 129/200] Avg loss: 3.3299
[20:56:27] [Epoch 130/200] Avg loss: 3.3186
[20:56:28] [Epoch 131/200] Avg loss: 3.2925
[20:56:30] [Epoch 132/200] Avg loss: 3.3148
[20:56:31] [Epoch 133/200] Avg loss: 3.3020
[20:56:33] [Epoch 134/200] Avg loss: 3.3024
[20:56:34] [Epoch 135/200] Avg loss: 3.2930
[20:56:36] [Epoch 136/200] Avg loss: 3.2931
[20:56:37] [Epoch 137/200] Avg loss: 3.2934
[20:56:39] [Epoch 138/200] Avg loss: 3.2903
[20:56:40] [Epoch 139/200] Avg loss: 3.2737
[20:56:50] [Epoch 140/200] Avg loss: 3.2553 | HR@10: 0.7647 NDCG@10: 0.5303
[20:56:52] [Epoch 141/200] Avg loss: 3.2509
[20:56:53] [Epoch 142/200] Avg loss: 3.2831
[20:56:55] [Epoch 143/200] Avg loss: 3.2545
[20:56:56] [Epoch 144/200] Avg loss: 3.2510
[20:56:58] [Epoch 145/200] Avg loss: 3.2466
[20:56:59] [Epoch 146/200] Avg loss: 3.2348
[20:57:01] [Epoch 147/200] Avg loss: 3.2279
[20:57:02] [Epoch 148/200] Avg loss: 3.2442
[20:57:04] [Epoch 149/200] Avg loss: 3.2145
[20:57:05] [Epoch 150/200] Avg loss: 3.2319
[20:57:07] [Epoch 151/200] Avg loss: 3.2143
[20:57:08] [Epoch 152/200] Avg loss: 3.2187
[20:57:10] [Epoch 153/200] Avg loss: 3.2148
[20:57:11] [Epoch 154/200] Avg loss: 3.2005
[20:57:13] [Epoch 155/200] Avg loss: 3.2019
[20:57:14] [Epoch 156/200] Avg loss: 3.2114
[20:57:16] [Epoch 157/200] Avg loss: 3.1897
[20:57:17] [Epoch 158/200] Avg loss: 3.1802
[20:57:19] [Epoch 159/200] Avg loss: 3.1958
[20:57:29] [Epoch 160/200] Avg loss: 3.2078 | HR@10: 0.7770 NDCG@10: 0.5442
[20:57:31] [Epoch 161/200] Avg loss: 3.1730
[20:57:32] [Epoch 162/200] Avg loss: 3.1730
[20:57:34] [Epoch 163/200] Avg loss: 3.1807
[20:57:35] [Epoch 164/200] Avg loss: 3.1889
[20:57:37] [Epoch 165/200] Avg loss: 3.1809
[20:57:38] [Epoch 166/200] Avg loss: 3.1811
[20:57:40] [Epoch 167/200] Avg loss: 3.1511
[20:57:41] [Epoch 168/200] Avg loss: 3.1678
[20:57:43] [Epoch 169/200] Avg loss: 3.1546
[20:57:45] [Epoch 170/200] Avg loss: 3.1541
[20:57:46] [Epoch 171/200] Avg loss: 3.1543
[20:57:48] [Epoch 172/200] Avg loss: 3.1675
[20:57:49] [Epoch 173/200] Avg loss: 3.1578
[20:57:51] [Epoch 174/200] Avg loss: 3.1363
[20:57:52] [Epoch 175/200] Avg loss: 3.1331
[20:57:54] [Epoch 176/200] Avg loss: 3.1058
[20:57:55] [Epoch 177/200] Avg loss: 3.1318
[20:57:57] [Epoch 178/200] Avg loss: 3.1411
[20:57:58] [Epoch 179/200] Avg loss: 3.1193
[20:58:08] [Epoch 180/200] Avg loss: 3.1325 | HR@10: 0.7778 NDCG@10: 0.5536
[20:58:10] [Epoch 181/200] Avg loss: 3.1349
[20:58:11] [Epoch 182/200] Avg loss: 3.1258
[20:58:13] [Epoch 183/200] Avg loss: 3.0964
[20:58:15] [Epoch 184/200] Avg loss: 3.1041
[20:58:16] [Epoch 185/200] Avg loss: 3.1158
[20:58:18] [Epoch 186/200] Avg loss: 3.1074
[20:58:19] [Epoch 187/200] Avg loss: 3.1099
[20:58:21] [Epoch 188/200] Avg loss: 3.1108
[20:58:22] [Epoch 189/200] Avg loss: 3.0937
[20:58:24] [Epoch 190/200] Avg loss: 3.0967
[20:58:25] [Epoch 191/200] Avg loss: 3.1108
[20:58:27] [Epoch 192/200] Avg loss: 3.0935
[20:58:28] [Epoch 193/200] Avg loss: 3.0958
[20:58:30] [Epoch 194/200] Avg loss: 3.0793
[20:58:31] [Epoch 195/200] Avg loss: 3.0717
[20:58:33] [Epoch 196/200] Avg loss: 3.0702
[20:58:35] [Epoch 197/200] Avg loss: 3.0642
[20:58:36] [Epoch 198/200] Avg loss: 3.0714
[20:58:38] [Epoch 199/200] Avg loss: 3.0659
[20:58:48] [Epoch 200/200] Avg loss: 3.0762 | HR@10: 0.7876 NDCG@10: 0.5654
[20:58:57] [Epochs 200] Test metrics: HR@10: 0.7586 NDCG@10: 0.5394

  Prediction demo (user 1):
    Recent history:  Toy Story (1995), Bug's Life, A (1998), Antz (1998), Hunchback of Notre Dame, The (1996), Hercules (1997)
    Ground truth:    Pocahontas (1995)
    Model top-10:
       1. Lion King, The (1994)
       2. Little Mermaid, The (1989)
       3. Jungle Book, The (1967)
       4. Anastasia (1997)
       5. South Park: Bigger, Longer and Uncut (1999)
       6. Sleeping Beauty (1959)
       7. Rescuers Down Under, The (1990)
       8. Beavis and Butt-head Do America (1996)
       9. Prince of Egypt, The (1998)
      10. Robin Hood (1973)
```

### Takeaway

**v2 Takeaways:**

**What we changed:** Only the loss function. The model architecture is identical to v1 — same SelfAttention, same AttentionBlock, same SASRec. The diff is in three places: sample_batch returns shared-per-user negatives (batch, num_negs) instead of per-position negatives (batch, seq_len), scoring uses matmul for the all-pairs negative computation, and BCE is replaced with sampled softmax.

**What sampled softmax does differently:** Instead of asking "is this item good? yes/no" independently for each candidate (BCE), sampled softmax asks "which of these 257 items is the correct next one?" The positive item must compete directly against 256 negatives in a single softmax distribution. Probabilities must sum to 1, so the model can't give everything a score of 0.99 — pushing one item up forces others down.

**Why the loss doesn't go to zero:** v1's BCE loss dropped from 5.3 to 0.59 and was still falling. v2's sampled softmax loss dropped from 17.9 to 3.07 and plateaued. This is correct and expected. A random model that assigns equal probability to all 257 candidates would have loss -log(1/257) ≈ 5.55. The model got down to 3.07, well below random but nowhere near zero — unlike v1's BCE which kept pushing toward zero.

**Why NDCG improved more than HR:** HR@10 asks a binary question — did the ground truth land in the top 10? v1 already got this right ~77% of the time. The bigger problem was *where* in the top 10 it landed. v1's overconfident scores meant the top 10 items had nearly identical scores, so the ground truth's position within the top 10 was essentially random. v2's better-separated scores push the ground truth toward rank 1-3 when it does appear, which is exactly what NDCG measures. The +0.054 NDCG gain versus +0.017 HR gain is the signature of reduced overconfidence.

**Why shared negatives work:** Each user's 256 negatives are reused across all 199 positions in their sequence. This works because an unseen item is unseen at every position — there's no position-specific reason to sample different negatives. v2 runs in comparable time to v1 (~6-7 min) despite scoring 256× more candidates.

**What's still wrong:** Sampled softmax with 256 negatives is better than BCE with 1, but it's still an approximation of full-catalog softmax (3,416 items). The model sees 256/3,416 ≈ 7.5% of the catalog per training step. And the Petrov & Macdonald research (2308.07192) showed that even sampled softmax exhibits some overconfidence — the predicted probabilities are still higher than the true item relevance probabilities. v3's gBCE solves this more elegantly: it goes back to BCE's simplicity but fixes the overconfidence directly by reweighting the positive/negative contributions to match the true catalog ratio. The punchline — gBCE with just 1 negative matches or beats SSM with 256 negatives, at the same training cost as v1.

## v3