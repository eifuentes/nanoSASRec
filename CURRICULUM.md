# nanoSASRec Walkthrough

Curriculum walkthrough from v1 to v8.

## Differences

Here are known differences from the original SASRec implementation.

- **Pre-LN normalization** (norm before attention/FFN). The original SASRec uses Post-LN (norm after residual add). Pre-LN is more stable for deeper models (Xiong et al. ICML 2020) and is what all modern implementations use, including HSTU and PinnerFormer.
- **Weight initialization** uses truncated normal with `std=1/sqrt(d)` (matching Meta's SASRec) instead of the original's Xavier uniform. Keeps initial dot products near zero for stable softmax in early training.
- **Reversed positional embeddings** (Huang 2025, arxiv: 2504.09596). The original uses forward indexing; v4+ counts backward from the prediction position so the model learns "how far back" rather than "what absolute slot."

These are intentional refinements, not bugs. v1-v3 use PyTorch default initialization and forward positions to match the original paper; v4+ applies these improvements.

## Reading the Metrics

**Avg loss (Sampled Softmax, v2/v4/v5/v6+):** The model ranks 1 correct item against 256 random negatives. Loss = -log(P) where P is the probability assigned to the correct item. The `~X% correct` in the training output is this probability converted back via e^(-loss).

| Loss | ~P(correct) | Meaning |
|------|-------------|---------|
| 5.5 | 0.4% | Random chance (1/257) |
| 3.0 | 5% | Weak signal |
| 2.0 | 14% | Learning |
| 1.0 | 37% | Strong |

**Avg loss (BCE, v1/v3):** Each position independently predicts positive (push toward 1) and negative (push toward 0). Loss near 0.69 is random (coin flip), lower is better. Not directly comparable to SSM loss values.

**HR@10 (Hit Rate):** Did the ground truth item land in the top 10 out of 101 candidates (1 true + 100 random negatives)? Binary per user, averaged across all users.

**NDCG@10 (Normalized Discounted Cumulative Gain):** How high in the top 10? Rewards placement: rank 1 = 1.0, rank 2 = 0.63, rank 5 = 0.39, rank 10 = 0.30. This is the metric that matters most — getting the right item to the top, not just into the list.

**Note:** Both HR and NDCG use 100 sampled negatives, matching the original SASRec protocol (Kang & McAuley 2018). This inflates absolute numbers vs full-catalog evaluation. Compare across versions within this repo, not against papers using full-catalog eval.

## v1

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
[03:40:09] [Epoch 21/200] Avg loss: 1.0583 | Avg P(pos): 0.6291
[03:40:11] [Epoch 22/200] Avg loss: 1.0544 | Avg P(pos): 0.6386
[03:40:13] [Epoch 23/200] Avg loss: 1.0490 | Avg P(pos): 0.6534
[03:40:14] [Epoch 24/200] Avg loss: 1.0383 | Avg P(pos): 0.6692
[03:40:16] [Epoch 25/200] Avg loss: 1.0390 | Avg P(pos): 0.6776
[03:40:18] [Epoch 26/200] Avg loss: 1.0330 | Avg P(pos): 0.6833
[03:40:20] [Epoch 27/200] Avg loss: 1.0295 | Avg P(pos): 0.6740
[03:40:22] [Epoch 28/200] Avg loss: 1.0258 | Avg P(pos): 0.6477
[03:40:24] [Epoch 29/200] Avg loss: 1.0236 | Avg P(pos): 0.6175
[03:40:25] [Epoch 30/200] Avg loss: 1.0145 | Avg P(pos): 0.5973
[03:40:27] [Epoch 31/200] Avg loss: 1.0162 | Avg P(pos): 0.5901
[03:40:29] [Epoch 32/200] Avg loss: 1.0094 | Avg P(pos): 0.5886
[03:40:31] [Epoch 33/200] Avg loss: 1.0128 | Avg P(pos): 0.5856
[03:40:33] [Epoch 34/200] Avg loss: 1.0081 | Avg P(pos): 0.5879
[03:40:34] [Epoch 35/200] Avg loss: 1.0043 | Avg P(pos): 0.5884
[03:40:36] [Epoch 36/200] Avg loss: 1.0089 | Avg P(pos): 0.5907
[03:40:38] [Epoch 37/200] Avg loss: 1.0029 | Avg P(pos): 0.5935
[03:40:40] [Epoch 38/200] Avg loss: 1.0006 | Avg P(pos): 0.5965
[03:40:42] [Epoch 39/200] Avg loss: 0.9997 | Avg P(pos): 0.6017
[03:40:52] [Epoch 40/200] Avg loss: 0.9954 | Avg P(pos): 0.6032 | HR@10: 0.4454 NDCG@10: 0.2419
[03:40:54] [Epoch 41/200] Avg loss: 0.9936 | Avg P(pos): 0.6026
[03:40:56] [Epoch 42/200] Avg loss: 0.9840 | Avg P(pos): 0.6059
[03:40:58] [Epoch 43/200] Avg loss: 0.9862 | Avg P(pos): 0.6072
[03:40:59] [Epoch 44/200] Avg loss: 0.9781 | Avg P(pos): 0.6083
[03:41:01] [Epoch 45/200] Avg loss: 0.9733 | Avg P(pos): 0.6065
[03:41:03] [Epoch 46/200] Avg loss: 0.9713 | Avg P(pos): 0.6052
[03:41:05] [Epoch 47/200] Avg loss: 0.9603 | Avg P(pos): 0.6038
[03:41:07] [Epoch 48/200] Avg loss: 0.9598 | Avg P(pos): 0.6038
[03:41:09] [Epoch 49/200] Avg loss: 0.9511 | Avg P(pos): 0.6018
[03:41:10] [Epoch 50/200] Avg loss: 0.9458 | Avg P(pos): 0.6003
[03:41:12] [Epoch 51/200] Avg loss: 0.9360 | Avg P(pos): 0.6013
[03:41:14] [Epoch 52/200] Avg loss: 0.9353 | Avg P(pos): 0.6019
[03:41:16] [Epoch 53/200] Avg loss: 0.9298 | Avg P(pos): 0.5994
[03:41:18] [Epoch 54/200] Avg loss: 0.9176 | Avg P(pos): 0.5985
[03:41:20] [Epoch 55/200] Avg loss: 0.9131 | Avg P(pos): 0.6001
[03:41:22] [Epoch 56/200] Avg loss: 0.9092 | Avg P(pos): 0.5985
[03:41:24] [Epoch 57/200] Avg loss: 0.9010 | Avg P(pos): 0.6003
[03:41:26] [Epoch 58/200] Avg loss: 0.9031 | Avg P(pos): 0.6012
[03:41:27] [Epoch 59/200] Avg loss: 0.8933 | Avg P(pos): 0.6025
[03:41:38] [Epoch 60/200] Avg loss: 0.8853 | Avg P(pos): 0.6050 | HR@10: 0.5399 NDCG@10: 0.3020
[03:41:40] [Epoch 61/200] Avg loss: 0.8837 | Avg P(pos): 0.6021
[03:41:42] [Epoch 62/200] Avg loss: 0.8835 | Avg P(pos): 0.6025
[03:41:44] [Epoch 63/200] Avg loss: 0.8777 | Avg P(pos): 0.6020
[03:41:45] [Epoch 64/200] Avg loss: 0.8758 | Avg P(pos): 0.5988
[03:41:47] [Epoch 65/200] Avg loss: 0.8692 | Avg P(pos): 0.6007
[03:41:49] [Epoch 66/200] Avg loss: 0.8654 | Avg P(pos): 0.6039
[03:41:51] [Epoch 67/200] Avg loss: 0.8625 | Avg P(pos): 0.6022
[03:41:53] [Epoch 68/200] Avg loss: 0.8592 | Avg P(pos): 0.5999
[03:41:55] [Epoch 69/200] Avg loss: 0.8522 | Avg P(pos): 0.6038
[03:41:57] [Epoch 70/200] Avg loss: 0.8527 | Avg P(pos): 0.5998
[03:41:59] [Epoch 71/200] Avg loss: 0.8466 | Avg P(pos): 0.6051
[03:42:01] [Epoch 72/200] Avg loss: 0.8395 | Avg P(pos): 0.6042
[03:42:03] [Epoch 73/200] Avg loss: 0.8390 | Avg P(pos): 0.6007
[03:42:05] [Epoch 74/200] Avg loss: 0.8386 | Avg P(pos): 0.6010
[03:42:06] [Epoch 75/200] Avg loss: 0.8339 | Avg P(pos): 0.6044
[03:42:08] [Epoch 76/200] Avg loss: 0.8323 | Avg P(pos): 0.6068
[03:42:10] [Epoch 77/200] Avg loss: 0.8218 | Avg P(pos): 0.6110
[03:42:12] [Epoch 78/200] Avg loss: 0.8183 | Avg P(pos): 0.6075
[03:42:14] [Epoch 79/200] Avg loss: 0.8139 | Avg P(pos): 0.6096
[03:42:24] [Epoch 80/200] Avg loss: 0.8148 | Avg P(pos): 0.6111 | HR@10: 0.6013 NDCG@10: 0.3481
[03:42:26] [Epoch 81/200] Avg loss: 0.8065 | Avg P(pos): 0.6123
[03:42:28] [Epoch 82/200] Avg loss: 0.8089 | Avg P(pos): 0.6101
[03:42:30] [Epoch 83/200] Avg loss: 0.7991 | Avg P(pos): 0.6091
[03:42:32] [Epoch 84/200] Avg loss: 0.7967 | Avg P(pos): 0.6113
[03:42:34] [Epoch 85/200] Avg loss: 0.7912 | Avg P(pos): 0.6102
[03:42:36] [Epoch 86/200] Avg loss: 0.7871 | Avg P(pos): 0.6133
[03:42:38] [Epoch 87/200] Avg loss: 0.7818 | Avg P(pos): 0.6135
[03:42:39] [Epoch 88/200] Avg loss: 0.7786 | Avg P(pos): 0.6155
[03:42:41] [Epoch 89/200] Avg loss: 0.7799 | Avg P(pos): 0.6080
[03:42:43] [Epoch 90/200] Avg loss: 0.7797 | Avg P(pos): 0.6112
[03:42:45] [Epoch 91/200] Avg loss: 0.7724 | Avg P(pos): 0.6125
[03:42:47] [Epoch 92/200] Avg loss: 0.7683 | Avg P(pos): 0.6151
[03:42:49] [Epoch 93/200] Avg loss: 0.7689 | Avg P(pos): 0.6140
[03:42:51] [Epoch 94/200] Avg loss: 0.7681 | Avg P(pos): 0.6135
[03:42:53] [Epoch 95/200] Avg loss: 0.7648 | Avg P(pos): 0.6116
[03:42:54] [Epoch 96/200] Avg loss: 0.7594 | Avg P(pos): 0.6156
[03:42:56] [Epoch 97/200] Avg loss: 0.7504 | Avg P(pos): 0.6190
[03:42:58] [Epoch 98/200] Avg loss: 0.7516 | Avg P(pos): 0.6139
[03:43:00] [Epoch 99/200] Avg loss: 0.7526 | Avg P(pos): 0.6135
[03:43:10] [Epoch 100/200] Avg loss: 0.7498 | Avg P(pos): 0.6140 | HR@10: 0.6627 NDCG@10: 0.3932
[03:43:12] [Epoch 101/200] Avg loss: 0.7504 | Avg P(pos): 0.6147
[03:43:14] [Epoch 102/200] Avg loss: 0.7442 | Avg P(pos): 0.6116
[03:43:16] [Epoch 103/200] Avg loss: 0.7456 | Avg P(pos): 0.6133
[03:43:18] [Epoch 104/200] Avg loss: 0.7427 | Avg P(pos): 0.6115
[03:43:20] [Epoch 105/200] Avg loss: 0.7450 | Avg P(pos): 0.6090
[03:43:22] [Epoch 106/200] Avg loss: 0.7383 | Avg P(pos): 0.6096
[03:43:24] [Epoch 107/200] Avg loss: 0.7302 | Avg P(pos): 0.6124
[03:43:25] [Epoch 108/200] Avg loss: 0.7300 | Avg P(pos): 0.6135
[03:43:27] [Epoch 109/200] Avg loss: 0.7297 | Avg P(pos): 0.6153
[03:43:29] [Epoch 110/200] Avg loss: 0.7314 | Avg P(pos): 0.6165
[03:43:31] [Epoch 111/200] Avg loss: 0.7239 | Avg P(pos): 0.6171
[03:43:33] [Epoch 112/200] Avg loss: 0.7288 | Avg P(pos): 0.6143
[03:43:35] [Epoch 113/200] Avg loss: 0.7249 | Avg P(pos): 0.6154
[03:43:37] [Epoch 114/200] Avg loss: 0.7215 | Avg P(pos): 0.6159
[03:43:39] [Epoch 115/200] Avg loss: 0.7221 | Avg P(pos): 0.6160
[03:43:41] [Epoch 116/200] Avg loss: 0.7124 | Avg P(pos): 0.6144
[03:43:43] [Epoch 117/200] Avg loss: 0.7128 | Avg P(pos): 0.6138
[03:43:44] [Epoch 118/200] Avg loss: 0.7108 | Avg P(pos): 0.6138
[03:43:46] [Epoch 119/200] Avg loss: 0.7092 | Avg P(pos): 0.6142
[03:43:57] [Epoch 120/200] Avg loss: 0.7109 | Avg P(pos): 0.6134 | HR@10: 0.6919 NDCG@10: 0.4178
[03:43:58] [Epoch 121/200] Avg loss: 0.7081 | Avg P(pos): 0.6142
[03:44:00] [Epoch 122/200] Avg loss: 0.7067 | Avg P(pos): 0.6155
[03:44:02] [Epoch 123/200] Avg loss: 0.7056 | Avg P(pos): 0.6134
[03:44:04] [Epoch 124/200] Avg loss: 0.7020 | Avg P(pos): 0.6199
[03:44:06] [Epoch 125/200] Avg loss: 0.6987 | Avg P(pos): 0.6240
[03:44:07] [Epoch 126/200] Avg loss: 0.7015 | Avg P(pos): 0.6178
[03:44:09] [Epoch 127/200] Avg loss: 0.6925 | Avg P(pos): 0.6174
[03:44:11] [Epoch 128/200] Avg loss: 0.6934 | Avg P(pos): 0.6202
[03:44:13] [Epoch 129/200] Avg loss: 0.6927 | Avg P(pos): 0.6197
[03:44:15] [Epoch 130/200] Avg loss: 0.6912 | Avg P(pos): 0.6250
[03:44:17] [Epoch 131/200] Avg loss: 0.6952 | Avg P(pos): 0.6188
[03:44:18] [Epoch 132/200] Avg loss: 0.6907 | Avg P(pos): 0.6233
[03:44:20] [Epoch 133/200] Avg loss: 0.6812 | Avg P(pos): 0.6243
[03:44:22] [Epoch 134/200] Avg loss: 0.6819 | Avg P(pos): 0.6267
[03:44:24] [Epoch 135/200] Avg loss: 0.6825 | Avg P(pos): 0.6267
[03:44:26] [Epoch 136/200] Avg loss: 0.6808 | Avg P(pos): 0.6256
[03:44:28] [Epoch 137/200] Avg loss: 0.6742 | Avg P(pos): 0.6279
[03:44:29] [Epoch 138/200] Avg loss: 0.6754 | Avg P(pos): 0.6304
[03:44:31] [Epoch 139/200] Avg loss: 0.6745 | Avg P(pos): 0.6279
[03:44:42] [Epoch 140/200] Avg loss: 0.6713 | Avg P(pos): 0.6269 | HR@10: 0.7139 NDCG@10: 0.4437
[03:44:44] [Epoch 141/200] Avg loss: 0.6715 | Avg P(pos): 0.6260
[03:44:45] [Epoch 142/200] Avg loss: 0.6703 | Avg P(pos): 0.6273
[03:44:47] [Epoch 143/200] Avg loss: 0.6676 | Avg P(pos): 0.6292
[03:44:49] [Epoch 144/200] Avg loss: 0.6659 | Avg P(pos): 0.6257
[03:44:51] [Epoch 145/200] Avg loss: 0.6593 | Avg P(pos): 0.6276
[03:44:53] [Epoch 146/200] Avg loss: 0.6628 | Avg P(pos): 0.6269
[03:44:55] [Epoch 147/200] Avg loss: 0.6572 | Avg P(pos): 0.6275
[03:44:57] [Epoch 148/200] Avg loss: 0.6564 | Avg P(pos): 0.6303
[03:44:59] [Epoch 149/200] Avg loss: 0.6577 | Avg P(pos): 0.6342
[03:45:01] [Epoch 150/200] Avg loss: 0.6568 | Avg P(pos): 0.6371
[03:45:02] [Epoch 151/200] Avg loss: 0.6587 | Avg P(pos): 0.6322
[03:45:04] [Epoch 152/200] Avg loss: 0.6548 | Avg P(pos): 0.6311
[03:45:06] [Epoch 153/200] Avg loss: 0.6555 | Avg P(pos): 0.6294
[03:45:08] [Epoch 154/200] Avg loss: 0.6502 | Avg P(pos): 0.6296
[03:45:10] [Epoch 155/200] Avg loss: 0.6457 | Avg P(pos): 0.6317
[03:45:12] [Epoch 156/200] Avg loss: 0.6480 | Avg P(pos): 0.6314
[03:45:14] [Epoch 157/200] Avg loss: 0.6420 | Avg P(pos): 0.6335
[03:45:16] [Epoch 158/200] Avg loss: 0.6448 | Avg P(pos): 0.6340
[03:45:18] [Epoch 159/200] Avg loss: 0.6453 | Avg P(pos): 0.6294
[03:45:28] [Epoch 160/200] Avg loss: 0.6383 | Avg P(pos): 0.6281 | HR@10: 0.7397 NDCG@10: 0.4668
[03:45:30] [Epoch 161/200] Avg loss: 0.6443 | Avg P(pos): 0.6279
[03:45:32] [Epoch 162/200] Avg loss: 0.6377 | Avg P(pos): 0.6282
[03:45:33] [Epoch 163/200] Avg loss: 0.6423 | Avg P(pos): 0.6325
[03:45:35] [Epoch 164/200] Avg loss: 0.6343 | Avg P(pos): 0.6338
[03:45:37] [Epoch 165/200] Avg loss: 0.6358 | Avg P(pos): 0.6249
[03:45:39] [Epoch 166/200] Avg loss: 0.6382 | Avg P(pos): 0.6278
[03:45:41] [Epoch 167/200] Avg loss: 0.6311 | Avg P(pos): 0.6326
[03:45:43] [Epoch 168/200] Avg loss: 0.6341 | Avg P(pos): 0.6305
[03:45:45] [Epoch 169/200] Avg loss: 0.6312 | Avg P(pos): 0.6309
[03:45:46] [Epoch 170/200] Avg loss: 0.6287 | Avg P(pos): 0.6331
[03:45:48] [Epoch 171/200] Avg loss: 0.6277 | Avg P(pos): 0.6340
[03:45:50] [Epoch 172/200] Avg loss: 0.6263 | Avg P(pos): 0.6344
[03:45:52] [Epoch 173/200] Avg loss: 0.6271 | Avg P(pos): 0.6336
[03:45:54] [Epoch 174/200] Avg loss: 0.6274 | Avg P(pos): 0.6381
[03:45:56] [Epoch 175/200] Avg loss: 0.6180 | Avg P(pos): 0.6336
[03:45:57] [Epoch 176/200] Avg loss: 0.6227 | Avg P(pos): 0.6294
[03:45:59] [Epoch 177/200] Avg loss: 0.6212 | Avg P(pos): 0.6362
[03:46:01] [Epoch 178/200] Avg loss: 0.6188 | Avg P(pos): 0.6386
[03:46:03] [Epoch 179/200] Avg loss: 0.6177 | Avg P(pos): 0.6332
[03:46:13] [Epoch 180/200] Avg loss: 0.6164 | Avg P(pos): 0.6375 | HR@10: 0.7568 NDCG@10: 0.4877
[03:46:15] [Epoch 181/200] Avg loss: 0.6201 | Avg P(pos): 0.6360
[03:46:17] [Epoch 182/200] Avg loss: 0.6128 | Avg P(pos): 0.6300
[03:46:19] [Epoch 183/200] Avg loss: 0.6141 | Avg P(pos): 0.6412
[03:46:20] [Epoch 184/200] Avg loss: 0.6178 | Avg P(pos): 0.6389
[03:46:22] [Epoch 185/200] Avg loss: 0.6167 | Avg P(pos): 0.6361
[03:46:24] [Epoch 186/200] Avg loss: 0.6110 | Avg P(pos): 0.6342
[03:46:26] [Epoch 187/200] Avg loss: 0.6104 | Avg P(pos): 0.6353
[03:46:28] [Epoch 188/200] Avg loss: 0.6122 | Avg P(pos): 0.6342
[03:46:29] [Epoch 189/200] Avg loss: 0.6135 | Avg P(pos): 0.6355
[03:46:31] [Epoch 190/200] Avg loss: 0.6106 | Avg P(pos): 0.6365
[03:46:33] [Epoch 191/200] Avg loss: 0.6099 | Avg P(pos): 0.6381
[03:46:35] [Epoch 192/200] Avg loss: 0.6090 | Avg P(pos): 0.6388
[03:46:37] [Epoch 193/200] Avg loss: 0.6038 | Avg P(pos): 0.6366
[03:46:39] [Epoch 194/200] Avg loss: 0.6065 | Avg P(pos): 0.6316
[03:46:41] [Epoch 195/200] Avg loss: 0.6095 | Avg P(pos): 0.6366
[03:46:42] [Epoch 196/200] Avg loss: 0.6017 | Avg P(pos): 0.6381
[03:46:44] [Epoch 197/200] Avg loss: 0.5984 | Avg P(pos): 0.6386
[03:46:46] [Epoch 198/200] Avg loss: 0.6034 | Avg P(pos): 0.6384
[03:46:48] [Epoch 199/200] Avg loss: 0.5996 | Avg P(pos): 0.6421
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
[04:11:07] [Epoch 21/200] Avg loss: 4.7408
[04:11:08] [Epoch 22/200] Avg loss: 4.7231
[04:11:09] [Epoch 23/200] Avg loss: 4.7119
[04:11:11] [Epoch 24/200] Avg loss: 4.7087
[04:11:12] [Epoch 25/200] Avg loss: 4.7064
[04:11:14] [Epoch 26/200] Avg loss: 4.6809
[04:11:15] [Epoch 27/200] Avg loss: 4.6807
[04:11:17] [Epoch 28/200] Avg loss: 4.6832
[04:11:18] [Epoch 29/200] Avg loss: 4.6674
[04:11:20] [Epoch 30/200] Avg loss: 4.6456
[04:11:22] [Epoch 31/200] Avg loss: 4.6475
[04:11:23] [Epoch 32/200] Avg loss: 4.6450
[04:11:25] [Epoch 33/200] Avg loss: 4.6289
[04:11:26] [Epoch 34/200] Avg loss: 4.6182
[04:11:28] [Epoch 35/200] Avg loss: 4.6261
[04:11:29] [Epoch 36/200] Avg loss: 4.6183
[04:11:31] [Epoch 37/200] Avg loss: 4.5954
[04:11:32] [Epoch 38/200] Avg loss: 4.5969
[04:11:34] [Epoch 39/200] Avg loss: 4.5760
[04:11:44] [Epoch 40/200] Avg loss: 4.5558 | HR@10: 0.4656 NDCG@10: 0.2591
[04:11:46] [Epoch 41/200] Avg loss: 4.5381
[04:11:47] [Epoch 42/200] Avg loss: 4.5200
[04:11:49] [Epoch 43/200] Avg loss: 4.4907
[04:11:50] [Epoch 44/200] Avg loss: 4.4573
[04:11:52] [Epoch 45/200] Avg loss: 4.4394
[04:11:53] [Epoch 46/200] Avg loss: 4.4167
[04:11:55] [Epoch 47/200] Avg loss: 4.3776
[04:11:56] [Epoch 48/200] Avg loss: 4.3461
[04:11:58] [Epoch 49/200] Avg loss: 4.3012
[04:11:59] [Epoch 50/200] Avg loss: 4.2836
[04:12:01] [Epoch 51/200] Avg loss: 4.2610
[04:12:02] [Epoch 52/200] Avg loss: 4.2209
[04:12:04] [Epoch 53/200] Avg loss: 4.2004
[04:12:05] [Epoch 54/200] Avg loss: 4.1652
[04:12:07] [Epoch 55/200] Avg loss: 4.1401
[04:12:08] [Epoch 56/200] Avg loss: 4.1393
[04:12:10] [Epoch 57/200] Avg loss: 4.1148
[04:12:11] [Epoch 58/200] Avg loss: 4.0929
[04:12:13] [Epoch 59/200] Avg loss: 4.0493
[04:12:23] [Epoch 60/200] Avg loss: 4.0385 | HR@10: 0.6321 NDCG@10: 0.3816
[04:12:24] [Epoch 61/200] Avg loss: 4.0112
[04:12:26] [Epoch 62/200] Avg loss: 4.0001
[04:12:27] [Epoch 63/200] Avg loss: 3.9930
[04:12:29] [Epoch 64/200] Avg loss: 3.9674
[04:12:30] [Epoch 65/200] Avg loss: 3.9392
[04:12:32] [Epoch 66/200] Avg loss: 3.9235
[04:12:33] [Epoch 67/200] Avg loss: 3.9107
[04:12:35] [Epoch 68/200] Avg loss: 3.8964
[04:12:37] [Epoch 69/200] Avg loss: 3.8656
[04:12:38] [Epoch 70/200] Avg loss: 3.8658
[04:12:40] [Epoch 71/200] Avg loss: 3.8364
[04:12:41] [Epoch 72/200] Avg loss: 3.8194
[04:12:43] [Epoch 73/200] Avg loss: 3.8049
[04:12:44] [Epoch 74/200] Avg loss: 3.7987
[04:12:46] [Epoch 75/200] Avg loss: 3.7887
[04:12:47] [Epoch 76/200] Avg loss: 3.7839
[04:12:49] [Epoch 77/200] Avg loss: 3.7496
[04:12:50] [Epoch 78/200] Avg loss: 3.7371
[04:12:52] [Epoch 79/200] Avg loss: 3.7302
[04:13:02] [Epoch 80/200] Avg loss: 3.7089 | HR@10: 0.6990 NDCG@10: 0.4449
[04:13:03] [Epoch 81/200] Avg loss: 3.6902
[04:13:05] [Epoch 82/200] Avg loss: 3.6815
[04:13:06] [Epoch 83/200] Avg loss: 3.6598
[04:13:08] [Epoch 84/200] Avg loss: 3.6666
[04:13:09] [Epoch 85/200] Avg loss: 3.6509
[04:13:11] [Epoch 86/200] Avg loss: 3.6123
[04:13:12] [Epoch 87/200] Avg loss: 3.6001
[04:13:14] [Epoch 88/200] Avg loss: 3.6037
[04:13:15] [Epoch 89/200] Avg loss: 3.5843
[04:13:17] [Epoch 90/200] Avg loss: 3.5715
[04:13:18] [Epoch 91/200] Avg loss: 3.5777
[04:13:20] [Epoch 92/200] Avg loss: 3.5376
[04:13:22] [Epoch 93/200] Avg loss: 3.5481
[04:13:23] [Epoch 94/200] Avg loss: 3.5286
[04:13:25] [Epoch 95/200] Avg loss: 3.5119
[04:13:26] [Epoch 96/200] Avg loss: 3.5259
[04:13:28] [Epoch 97/200] Avg loss: 3.5099
[04:13:29] [Epoch 98/200] Avg loss: 3.4976
[04:13:31] [Epoch 99/200] Avg loss: 3.5153
[04:13:41] [Epoch 100/200] Avg loss: 3.4710 | HR@10: 0.7361 NDCG@10: 0.4869
[04:13:43] [Epoch 101/200] Avg loss: 3.4916
[04:13:44] [Epoch 102/200] Avg loss: 3.4577
[04:13:46] [Epoch 103/200] Avg loss: 3.4612
[04:13:47] [Epoch 104/200] Avg loss: 3.4383
[04:13:49] [Epoch 105/200] Avg loss: 3.4438
[04:13:50] [Epoch 106/200] Avg loss: 3.4546
[04:13:52] [Epoch 107/200] Avg loss: 3.4403
[04:13:53] [Epoch 108/200] Avg loss: 3.4192
[04:13:55] [Epoch 109/200] Avg loss: 3.4128
[04:13:56] [Epoch 110/200] Avg loss: 3.4092
[04:13:58] [Epoch 111/200] Avg loss: 3.3777
[04:13:59] [Epoch 112/200] Avg loss: 3.4009
[04:14:01] [Epoch 113/200] Avg loss: 3.3779
[04:14:02] [Epoch 114/200] Avg loss: 3.3740
[04:14:04] [Epoch 115/200] Avg loss: 3.3739
[04:14:05] [Epoch 116/200] Avg loss: 3.3744
[04:14:07] [Epoch 117/200] Avg loss: 3.3573
[04:14:08] [Epoch 118/200] Avg loss: 3.3407
[04:14:10] [Epoch 119/200] Avg loss: 3.3385
[04:14:20] [Epoch 120/200] Avg loss: 3.3255 | HR@10: 0.7565 NDCG@10: 0.5151
[04:14:22] [Epoch 121/200] Avg loss: 3.3405
[04:14:23] [Epoch 122/200] Avg loss: 3.3313
[04:14:24] [Epoch 123/200] Avg loss: 3.3413
[04:14:26] [Epoch 124/200] Avg loss: 3.2930
[04:14:27] [Epoch 125/200] Avg loss: 3.3246
[04:14:29] [Epoch 126/200] Avg loss: 3.2860
[04:14:30] [Epoch 127/200] Avg loss: 3.2672
[04:14:32] [Epoch 128/200] Avg loss: 3.2991
[04:14:33] [Epoch 129/200] Avg loss: 3.2869
[04:14:35] [Epoch 130/200] Avg loss: 3.2717
[04:14:36] [Epoch 131/200] Avg loss: 3.2807
[04:14:38] [Epoch 132/200] Avg loss: 3.2521
[04:14:39] [Epoch 133/200] Avg loss: 3.2591
[04:14:41] [Epoch 134/200] Avg loss: 3.2377
[04:14:42] [Epoch 135/200] Avg loss: 3.2258
[04:14:44] [Epoch 136/200] Avg loss: 3.2460
[04:14:45] [Epoch 137/200] Avg loss: 3.2485
[04:14:47] [Epoch 138/200] Avg loss: 3.2446
[04:14:48] [Epoch 139/200] Avg loss: 3.2298
[04:14:59] [Epoch 140/200] Avg loss: 3.2366 | HR@10: 0.7684 NDCG@10: 0.5361
[04:15:00] [Epoch 141/200] Avg loss: 3.2247
[04:15:02] [Epoch 142/200] Avg loss: 3.2199
[04:15:03] [Epoch 143/200] Avg loss: 3.2211
[04:15:05] [Epoch 144/200] Avg loss: 3.2199
[04:15:06] [Epoch 145/200] Avg loss: 3.1956
[04:15:08] [Epoch 146/200] Avg loss: 3.2177
[04:15:09] [Epoch 147/200] Avg loss: 3.1765
[04:15:11] [Epoch 148/200] Avg loss: 3.1574
[04:15:12] [Epoch 149/200] Avg loss: 3.1921
[04:15:14] [Epoch 150/200] Avg loss: 3.1877
[04:15:15] [Epoch 151/200] Avg loss: 3.1781
[04:15:17] [Epoch 152/200] Avg loss: 3.1987
[04:15:18] [Epoch 153/200] Avg loss: 3.1891
[04:15:20] [Epoch 154/200] Avg loss: 3.1687
[04:15:21] [Epoch 155/200] Avg loss: 3.1534
[04:15:23] [Epoch 156/200] Avg loss: 3.1677
[04:15:24] [Epoch 157/200] Avg loss: 3.1659
[04:15:26] [Epoch 158/200] Avg loss: 3.1421
[04:15:27] [Epoch 159/200] Avg loss: 3.1377
[04:15:37] [Epoch 160/200] Avg loss: 3.1603 | HR@10: 0.7775 NDCG@10: 0.5487
[04:15:39] [Epoch 161/200] Avg loss: 3.1602
[04:15:40] [Epoch 162/200] Avg loss: 3.1231
[04:15:42] [Epoch 163/200] Avg loss: 3.1450
[04:15:43] [Epoch 164/200] Avg loss: 3.1258
[04:15:45] [Epoch 165/200] Avg loss: 3.1158
[04:15:47] [Epoch 166/200] Avg loss: 3.1225
[04:15:48] [Epoch 167/200] Avg loss: 3.1357
[04:15:50] [Epoch 168/200] Avg loss: 3.1107
[04:15:51] [Epoch 169/200] Avg loss: 3.1031
[04:15:53] [Epoch 170/200] Avg loss: 3.1216
[04:15:54] [Epoch 171/200] Avg loss: 3.1107
[04:15:56] [Epoch 172/200] Avg loss: 3.1286
[04:15:57] [Epoch 173/200] Avg loss: 3.1028
[04:15:59] [Epoch 174/200] Avg loss: 3.0871
[04:16:00] [Epoch 175/200] Avg loss: 3.0937
[04:16:02] [Epoch 176/200] Avg loss: 3.1017
[04:16:03] [Epoch 177/200] Avg loss: 3.0807
[04:16:05] [Epoch 178/200] Avg loss: 3.0878
[04:16:06] [Epoch 179/200] Avg loss: 3.0819
[04:16:17] [Epoch 180/200] Avg loss: 3.0888 | HR@10: 0.7866 NDCG@10: 0.5566
[04:16:19] [Epoch 181/200] Avg loss: 3.0754
[04:16:20] [Epoch 182/200] Avg loss: 3.1015
[04:16:22] [Epoch 183/200] Avg loss: 3.0854
[04:16:23] [Epoch 184/200] Avg loss: 3.0645
[04:16:25] [Epoch 185/200] Avg loss: 3.0780
[04:16:26] [Epoch 186/200] Avg loss: 3.0653
[04:16:28] [Epoch 187/200] Avg loss: 3.0654
[04:16:29] [Epoch 188/200] Avg loss: 3.0732
[04:16:31] [Epoch 189/200] Avg loss: 3.0498
[04:16:33] [Epoch 190/200] Avg loss: 3.0659
[04:16:34] [Epoch 191/200] Avg loss: 3.0477
[04:16:36] [Epoch 192/200] Avg loss: 3.0481
[04:16:37] [Epoch 193/200] Avg loss: 3.0375
[04:16:39] [Epoch 194/200] Avg loss: 3.0498
[04:16:40] [Epoch 195/200] Avg loss: 3.0386
[04:16:42] [Epoch 196/200] Avg loss: 3.0328
[04:16:43] [Epoch 197/200] Avg loss: 3.0486
[04:16:45] [Epoch 198/200] Avg loss: 3.0548
[04:16:46] [Epoch 199/200] Avg loss: 3.0458
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

**v2 Takeaways:**

**What we changed:** Only the loss function. The model architecture is identical to v1 - same SelfAttention, same AttentionBlock, same SASRec. The diff is in three places: sample_batch returns shared-per-user negatives (batch, num_negs) instead of per-position negatives (batch, seq_len), scoring uses matmul for the all-pairs negative computation, and BCE is replaced with sampled softmax.

**What sampled softmax does differently:** Instead of asking "is this item good? yes/no" independently for each candidate (BCE), sampled softmax asks "which of these 257 items is the correct next one?" The positive item must compete directly against 256 negatives in a single softmax distribution. Probabilities must sum to 1, so the model can't give everything a score of 0.99 - pushing one item up forces others down.

**Why the loss doesn't go to zero:** v1's BCE loss dropped from 5.3 to 0.59 and was still falling. v2's sampled softmax loss dropped from 17.9 to 3.07 and plateaued. This is correct and expected. A random model that assigns equal probability to all 257 candidates would have loss -log(1/257) ≈ 5.55. The model got down to 3.07, well below random but nowhere near zero - unlike v1's BCE which kept pushing toward zero.

**Why NDCG improved more than HR:** HR@10 asks a binary question - did the ground truth land in the top 10? v1 already got this right ~77% of the time. The bigger problem was *where* in the top 10 it landed. v1's overconfident scores meant the top 10 items had nearly identical scores, so the ground truth's position within the top 10 was essentially random. v2's better-separated scores push the ground truth toward rank 1-3 when it does appear, which is exactly what NDCG measures. The +0.054 NDCG gain versus +0.017 HR gain is the signature of reduced overconfidence.

**Why shared negatives work:** Each user's 256 negatives are reused across all 199 positions in their sequence. This works because an unseen item is unseen at every position - there's no position-specific reason to sample different negatives. v2 runs in comparable time to v1 (~6-7 min) despite scoring 256× more candidates.

**What's still wrong:** Sampled softmax with 256 negatives is better than BCE with 1, but it's still an approximation of full-catalog softmax (3,416 items). The model sees 256/3,416 ≈ 7.5% of the catalog per training step. And the Petrov & Macdonald research (2308.07192) showed that even sampled softmax exhibits some overconfidence - the predicted probabilities are still higher than the true item relevance probabilities. v3's gBCE solves this more elegantly: it goes back to BCE's simplicity but fixes the overconfidence directly by reweighting the positive/negative contributions to match the true catalog ratio. The punchline - gBCE with just 1 negative matches or beats SSM with 256 negatives, at the same training cost as v1.

## v3

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
[13:57:37] [Epoch 21/200] Avg loss: 0.5137 | Avg P(pos): 0.6277
[13:57:39] [Epoch 22/200] Avg loss: 0.5116 | Avg P(pos): 0.6388
[13:57:41] [Epoch 23/200] Avg loss: 0.5087 | Avg P(pos): 0.6489
[13:57:42] [Epoch 24/200] Avg loss: 0.5061 | Avg P(pos): 0.6527
[13:57:44] [Epoch 25/200] Avg loss: 0.5046 | Avg P(pos): 0.6607
[13:57:46] [Epoch 26/200] Avg loss: 0.5037 | Avg P(pos): 0.6546
[13:57:48] [Epoch 27/200] Avg loss: 0.5019 | Avg P(pos): 0.6380
[13:57:49] [Epoch 28/200] Avg loss: 0.4993 | Avg P(pos): 0.6352
[13:57:51] [Epoch 29/200] Avg loss: 0.4992 | Avg P(pos): 0.6291
[13:57:53] [Epoch 30/200] Avg loss: 0.4968 | Avg P(pos): 0.6296
[13:57:55] [Epoch 31/200] Avg loss: 0.4957 | Avg P(pos): 0.6299
[13:57:56] [Epoch 32/200] Avg loss: 0.4949 | Avg P(pos): 0.6329
[13:57:58] [Epoch 33/200] Avg loss: 0.4942 | Avg P(pos): 0.6346
[13:58:00] [Epoch 34/200] Avg loss: 0.4903 | Avg P(pos): 0.6361
[13:58:02] [Epoch 35/200] Avg loss: 0.4900 | Avg P(pos): 0.6353
[13:58:03] [Epoch 36/200] Avg loss: 0.4884 | Avg P(pos): 0.6364
[13:58:05] [Epoch 37/200] Avg loss: 0.4867 | Avg P(pos): 0.6386
[13:58:07] [Epoch 38/200] Avg loss: 0.4804 | Avg P(pos): 0.6441
[13:58:09] [Epoch 39/200] Avg loss: 0.4828 | Avg P(pos): 0.6406
[13:58:19] [Epoch 40/200] Avg loss: 0.4773 | Avg P(pos): 0.6448 | HR@10: 0.4623 NDCG@10: 0.2557
[13:58:21] [Epoch 41/200] Avg loss: 0.4759 | Avg P(pos): 0.6446
[13:58:23] [Epoch 42/200] Avg loss: 0.4703 | Avg P(pos): 0.6476
[13:58:24] [Epoch 43/200] Avg loss: 0.4682 | Avg P(pos): 0.6509
[13:58:26] [Epoch 44/200] Avg loss: 0.4653 | Avg P(pos): 0.6532
[13:58:28] [Epoch 45/200] Avg loss: 0.4602 | Avg P(pos): 0.6566
[13:58:30] [Epoch 46/200] Avg loss: 0.4583 | Avg P(pos): 0.6584
[13:58:31] [Epoch 47/200] Avg loss: 0.4546 | Avg P(pos): 0.6599
[13:58:33] [Epoch 48/200] Avg loss: 0.4538 | Avg P(pos): 0.6600
[13:58:35] [Epoch 49/200] Avg loss: 0.4490 | Avg P(pos): 0.6632
[13:58:37] [Epoch 50/200] Avg loss: 0.4435 | Avg P(pos): 0.6630
[13:58:38] [Epoch 51/200] Avg loss: 0.4421 | Avg P(pos): 0.6640
[13:58:40] [Epoch 52/200] Avg loss: 0.4392 | Avg P(pos): 0.6667
[13:58:42] [Epoch 53/200] Avg loss: 0.4347 | Avg P(pos): 0.6689
[13:58:44] [Epoch 54/200] Avg loss: 0.4361 | Avg P(pos): 0.6686
[13:58:46] [Epoch 55/200] Avg loss: 0.4335 | Avg P(pos): 0.6713
[13:58:47] [Epoch 56/200] Avg loss: 0.4304 | Avg P(pos): 0.6710
[13:58:49] [Epoch 57/200] Avg loss: 0.4257 | Avg P(pos): 0.6726
[13:58:51] [Epoch 58/200] Avg loss: 0.4263 | Avg P(pos): 0.6709
[13:58:53] [Epoch 59/200] Avg loss: 0.4224 | Avg P(pos): 0.6728
[13:59:03] [Epoch 60/200] Avg loss: 0.4228 | Avg P(pos): 0.6717 | HR@10: 0.5616 NDCG@10: 0.3209
[13:59:05] [Epoch 61/200] Avg loss: 0.4203 | Avg P(pos): 0.6698
[13:59:07] [Epoch 62/200] Avg loss: 0.4190 | Avg P(pos): 0.6690
[13:59:08] [Epoch 63/200] Avg loss: 0.4151 | Avg P(pos): 0.6702
[13:59:10] [Epoch 64/200] Avg loss: 0.4129 | Avg P(pos): 0.6696
[13:59:12] [Epoch 65/200] Avg loss: 0.4116 | Avg P(pos): 0.6659
[13:59:14] [Epoch 66/200] Avg loss: 0.4089 | Avg P(pos): 0.6651
[13:59:15] [Epoch 67/200] Avg loss: 0.4059 | Avg P(pos): 0.6622
[13:59:17] [Epoch 68/200] Avg loss: 0.4042 | Avg P(pos): 0.6580
[13:59:19] [Epoch 69/200] Avg loss: 0.4003 | Avg P(pos): 0.6587
[13:59:21] [Epoch 70/200] Avg loss: 0.4002 | Avg P(pos): 0.6543
[13:59:22] [Epoch 71/200] Avg loss: 0.3959 | Avg P(pos): 0.6592
[13:59:24] [Epoch 72/200] Avg loss: 0.3967 | Avg P(pos): 0.6563
[13:59:26] [Epoch 73/200] Avg loss: 0.3922 | Avg P(pos): 0.6548
[13:59:28] [Epoch 74/200] Avg loss: 0.3889 | Avg P(pos): 0.6560
[13:59:29] [Epoch 75/200] Avg loss: 0.3888 | Avg P(pos): 0.6557
[13:59:31] [Epoch 76/200] Avg loss: 0.3875 | Avg P(pos): 0.6535
[13:59:33] [Epoch 77/200] Avg loss: 0.3860 | Avg P(pos): 0.6546
[13:59:35] [Epoch 78/200] Avg loss: 0.3828 | Avg P(pos): 0.6572
[13:59:36] [Epoch 79/200] Avg loss: 0.3772 | Avg P(pos): 0.6584
[13:59:47] [Epoch 80/200] Avg loss: 0.3787 | Avg P(pos): 0.6594 | HR@10: 0.6513 NDCG@10: 0.3895
[13:59:49] [Epoch 81/200] Avg loss: 0.3768 | Avg P(pos): 0.6583
[13:59:51] [Epoch 82/200] Avg loss: 0.3731 | Avg P(pos): 0.6598
[13:59:52] [Epoch 83/200] Avg loss: 0.3739 | Avg P(pos): 0.6586
[13:59:54] [Epoch 84/200] Avg loss: 0.3688 | Avg P(pos): 0.6601
[13:59:56] [Epoch 85/200] Avg loss: 0.3670 | Avg P(pos): 0.6620
[13:59:58] [Epoch 86/200] Avg loss: 0.3669 | Avg P(pos): 0.6579
[14:00:00] [Epoch 87/200] Avg loss: 0.3654 | Avg P(pos): 0.6630
[14:00:01] [Epoch 88/200] Avg loss: 0.3609 | Avg P(pos): 0.6615
[14:00:03] [Epoch 89/200] Avg loss: 0.3609 | Avg P(pos): 0.6632
[14:00:05] [Epoch 90/200] Avg loss: 0.3610 | Avg P(pos): 0.6630
[14:00:07] [Epoch 91/200] Avg loss: 0.3584 | Avg P(pos): 0.6634
[14:00:08] [Epoch 92/200] Avg loss: 0.3573 | Avg P(pos): 0.6677
[14:00:10] [Epoch 93/200] Avg loss: 0.3564 | Avg P(pos): 0.6663
[14:00:12] [Epoch 94/200] Avg loss: 0.3535 | Avg P(pos): 0.6652
[14:00:14] [Epoch 95/200] Avg loss: 0.3510 | Avg P(pos): 0.6672
[14:00:15] [Epoch 96/200] Avg loss: 0.3499 | Avg P(pos): 0.6680
[14:00:17] [Epoch 97/200] Avg loss: 0.3482 | Avg P(pos): 0.6674
[14:00:19] [Epoch 98/200] Avg loss: 0.3468 | Avg P(pos): 0.6705
[14:00:21] [Epoch 99/200] Avg loss: 0.3441 | Avg P(pos): 0.6721
[14:00:31] [Epoch 100/200] Avg loss: 0.3452 | Avg P(pos): 0.6703 | HR@10: 0.7048 NDCG@10: 0.4339
[14:00:33] [Epoch 101/200] Avg loss: 0.3435 | Avg P(pos): 0.6666
[14:00:35] [Epoch 102/200] Avg loss: 0.3410 | Avg P(pos): 0.6686
[14:00:37] [Epoch 103/200] Avg loss: 0.3401 | Avg P(pos): 0.6692
[14:00:39] [Epoch 104/200] Avg loss: 0.3405 | Avg P(pos): 0.6703
[14:00:41] [Epoch 105/200] Avg loss: 0.3364 | Avg P(pos): 0.6726
[14:00:43] [Epoch 106/200] Avg loss: 0.3350 | Avg P(pos): 0.6702
[14:00:45] [Epoch 107/200] Avg loss: 0.3356 | Avg P(pos): 0.6753
[14:00:47] [Epoch 108/200] Avg loss: 0.3345 | Avg P(pos): 0.6714
[14:00:48] [Epoch 109/200] Avg loss: 0.3326 | Avg P(pos): 0.6723
[14:00:50] [Epoch 110/200] Avg loss: 0.3328 | Avg P(pos): 0.6729
[14:00:52] [Epoch 111/200] Avg loss: 0.3283 | Avg P(pos): 0.6729
[14:00:54] [Epoch 112/200] Avg loss: 0.3284 | Avg P(pos): 0.6729
[14:00:55] [Epoch 113/200] Avg loss: 0.3283 | Avg P(pos): 0.6763
[14:00:57] [Epoch 114/200] Avg loss: 0.3285 | Avg P(pos): 0.6739
[14:00:59] [Epoch 115/200] Avg loss: 0.3233 | Avg P(pos): 0.6774
[14:01:01] [Epoch 116/200] Avg loss: 0.3250 | Avg P(pos): 0.6738
[14:01:02] [Epoch 117/200] Avg loss: 0.3211 | Avg P(pos): 0.6774
[14:01:04] [Epoch 118/200] Avg loss: 0.3227 | Avg P(pos): 0.6778
[14:01:06] [Epoch 119/200] Avg loss: 0.3202 | Avg P(pos): 0.6798
[14:01:17] [Epoch 120/200] Avg loss: 0.3219 | Avg P(pos): 0.6806 | HR@10: 0.7439 NDCG@10: 0.4697
[14:01:19] [Epoch 121/200] Avg loss: 0.3201 | Avg P(pos): 0.6827
[14:01:21] [Epoch 122/200] Avg loss: 0.3185 | Avg P(pos): 0.6847
[14:01:23] [Epoch 123/200] Avg loss: 0.3173 | Avg P(pos): 0.6833
[14:01:25] [Epoch 124/200] Avg loss: 0.3173 | Avg P(pos): 0.6802
[14:01:26] [Epoch 125/200] Avg loss: 0.3194 | Avg P(pos): 0.6787
[14:01:28] [Epoch 126/200] Avg loss: 0.3140 | Avg P(pos): 0.6822
[14:01:30] [Epoch 127/200] Avg loss: 0.3123 | Avg P(pos): 0.6848
[14:01:32] [Epoch 128/200] Avg loss: 0.3125 | Avg P(pos): 0.6872
[14:01:33] [Epoch 129/200] Avg loss: 0.3090 | Avg P(pos): 0.6886
[14:01:35] [Epoch 130/200] Avg loss: 0.3131 | Avg P(pos): 0.6852
[14:01:37] [Epoch 131/200] Avg loss: 0.3105 | Avg P(pos): 0.6905
[14:01:39] [Epoch 132/200] Avg loss: 0.3095 | Avg P(pos): 0.6908
[14:01:41] [Epoch 133/200] Avg loss: 0.3075 | Avg P(pos): 0.6898
[14:01:43] [Epoch 134/200] Avg loss: 0.3081 | Avg P(pos): 0.6885
[14:01:44] [Epoch 135/200] Avg loss: 0.3093 | Avg P(pos): 0.6899
[14:01:46] [Epoch 136/200] Avg loss: 0.3058 | Avg P(pos): 0.6950
[14:01:48] [Epoch 137/200] Avg loss: 0.3052 | Avg P(pos): 0.6940
[14:01:50] [Epoch 138/200] Avg loss: 0.3041 | Avg P(pos): 0.6966
[14:01:52] [Epoch 139/200] Avg loss: 0.3051 | Avg P(pos): 0.6963
[14:02:02] [Epoch 140/200] Avg loss: 0.3041 | Avg P(pos): 0.6968 | HR@10: 0.7641 NDCG@10: 0.4890
[14:02:04] [Epoch 141/200] Avg loss: 0.3042 | Avg P(pos): 0.6970
[14:02:06] [Epoch 142/200] Avg loss: 0.3051 | Avg P(pos): 0.6982
[14:02:07] [Epoch 143/200] Avg loss: 0.3015 | Avg P(pos): 0.6974
[14:02:09] [Epoch 144/200] Avg loss: 0.3045 | Avg P(pos): 0.6976
[14:02:11] [Epoch 145/200] Avg loss: 0.3012 | Avg P(pos): 0.6997
[14:02:13] [Epoch 146/200] Avg loss: 0.2995 | Avg P(pos): 0.7012
[14:02:14] [Epoch 147/200] Avg loss: 0.2991 | Avg P(pos): 0.7004
[14:02:16] [Epoch 148/200] Avg loss: 0.3016 | Avg P(pos): 0.6997
[14:02:18] [Epoch 149/200] Avg loss: 0.2971 | Avg P(pos): 0.7006
[14:02:20] [Epoch 150/200] Avg loss: 0.2973 | Avg P(pos): 0.7012
[14:02:22] [Epoch 151/200] Avg loss: 0.2965 | Avg P(pos): 0.7025
[14:02:23] [Epoch 152/200] Avg loss: 0.2972 | Avg P(pos): 0.7033
[14:02:25] [Epoch 153/200] Avg loss: 0.2957 | Avg P(pos): 0.7010
[14:02:27] [Epoch 154/200] Avg loss: 0.2954 | Avg P(pos): 0.7033
[14:02:29] [Epoch 155/200] Avg loss: 0.2933 | Avg P(pos): 0.7021
[14:02:31] [Epoch 156/200] Avg loss: 0.2951 | Avg P(pos): 0.7026
[14:02:32] [Epoch 157/200] Avg loss: 0.2940 | Avg P(pos): 0.7053
[14:02:34] [Epoch 158/200] Avg loss: 0.2932 | Avg P(pos): 0.7050
[14:02:36] [Epoch 159/200] Avg loss: 0.2923 | Avg P(pos): 0.7038
[14:02:48] [Epoch 160/200] Avg loss: 0.2957 | Avg P(pos): 0.7060 | HR@10: 0.7720 NDCG@10: 0.5051
[14:02:50] [Epoch 161/200] Avg loss: 0.2937 | Avg P(pos): 0.7002
[14:02:52] [Epoch 162/200] Avg loss: 0.2906 | Avg P(pos): 0.7049
[14:02:54] [Epoch 163/200] Avg loss: 0.2902 | Avg P(pos): 0.7071
[14:02:56] [Epoch 164/200] Avg loss: 0.2900 | Avg P(pos): 0.7057
[14:02:58] [Epoch 165/200] Avg loss: 0.2897 | Avg P(pos): 0.7090
[14:03:00] [Epoch 166/200] Avg loss: 0.2915 | Avg P(pos): 0.7096
[14:03:02] [Epoch 167/200] Avg loss: 0.2912 | Avg P(pos): 0.7103
[14:03:04] [Epoch 168/200] Avg loss: 0.2885 | Avg P(pos): 0.7093
[14:03:06] [Epoch 169/200] Avg loss: 0.2883 | Avg P(pos): 0.7121
[14:03:08] [Epoch 170/200] Avg loss: 0.2887 | Avg P(pos): 0.7134
[14:03:10] [Epoch 171/200] Avg loss: 0.2860 | Avg P(pos): 0.7152
[14:03:12] [Epoch 172/200] Avg loss: 0.2903 | Avg P(pos): 0.7104
[14:03:14] [Epoch 173/200] Avg loss: 0.2888 | Avg P(pos): 0.7136
[14:03:15] [Epoch 174/200] Avg loss: 0.2855 | Avg P(pos): 0.7143
[14:03:17] [Epoch 175/200] Avg loss: 0.2855 | Avg P(pos): 0.7135
[14:03:19] [Epoch 176/200] Avg loss: 0.2863 | Avg P(pos): 0.7115
[14:03:21] [Epoch 177/200] Avg loss: 0.2832 | Avg P(pos): 0.7136
[14:03:23] [Epoch 178/200] Avg loss: 0.2873 | Avg P(pos): 0.7163
[14:03:25] [Epoch 179/200] Avg loss: 0.2832 | Avg P(pos): 0.7183
[14:03:36] [Epoch 180/200] Avg loss: 0.2861 | Avg P(pos): 0.7138 | HR@10: 0.7844 NDCG@10: 0.5205
[14:03:38] [Epoch 181/200] Avg loss: 0.2832 | Avg P(pos): 0.7227
[14:03:40] [Epoch 182/200] Avg loss: 0.2838 | Avg P(pos): 0.7217
[14:03:42] [Epoch 183/200] Avg loss: 0.2815 | Avg P(pos): 0.7212
[14:03:44] [Epoch 184/200] Avg loss: 0.2828 | Avg P(pos): 0.7212
[14:03:46] [Epoch 185/200] Avg loss: 0.2829 | Avg P(pos): 0.7194
[14:03:48] [Epoch 186/200] Avg loss: 0.2796 | Avg P(pos): 0.7228
[14:03:50] [Epoch 187/200] Avg loss: 0.2812 | Avg P(pos): 0.7198
[14:03:52] [Epoch 188/200] Avg loss: 0.2793 | Avg P(pos): 0.7193
[14:03:54] [Epoch 189/200] Avg loss: 0.2788 | Avg P(pos): 0.7209
[14:03:56] [Epoch 190/200] Avg loss: 0.2805 | Avg P(pos): 0.7234
[14:03:58] [Epoch 191/200] Avg loss: 0.2809 | Avg P(pos): 0.7247
[14:04:00] [Epoch 192/200] Avg loss: 0.2824 | Avg P(pos): 0.7249
[14:04:02] [Epoch 193/200] Avg loss: 0.2776 | Avg P(pos): 0.7258
[14:04:04] [Epoch 194/200] Avg loss: 0.2799 | Avg P(pos): 0.7259
[14:04:06] [Epoch 195/200] Avg loss: 0.2782 | Avg P(pos): 0.7295
[14:04:08] [Epoch 196/200] Avg loss: 0.2804 | Avg P(pos): 0.7287
[14:04:10] [Epoch 197/200] Avg loss: 0.2777 | Avg P(pos): 0.7293
[14:04:12] [Epoch 198/200] Avg loss: 0.2781 | Avg P(pos): 0.7325
[14:04:14] [Epoch 199/200] Avg loss: 0.2779 | Avg P(pos): 0.7284
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

## v4

### Run 

`uv run python train_v4.py --device mps --epochs 200 --eval-every 20`

### Output

```bash
➜  nanoSASRec git:(feat--train-v4) ✗ uv run python train_v4.py --device mps --epochs 200 --eval-every 20
Number Users: 6,040
Number Items: 3,416
Dataset Lengths: 6040 6040 6040
Model parameters: 211,500
Estimate batches per epoch: 47
[04:11:36] [Epoch 1/200] Avg loss: 5.3592
[04:11:38] [Epoch 2/200] Avg loss: 4.7685
[04:11:40] [Epoch 3/200] Avg loss: 4.7217
[04:11:41] [Epoch 4/200] Avg loss: 4.7038
[04:11:43] [Epoch 5/200] Avg loss: 4.7042
[04:11:45] [Epoch 6/200] Avg loss: 4.6434
[04:11:46] [Epoch 7/200] Avg loss: 4.5344
[04:11:48] [Epoch 8/200] Avg loss: 4.4006
[04:11:50] [Epoch 9/200] Avg loss: 4.2957
[04:11:52] [Epoch 10/200] Avg loss: 4.1981
[04:11:53] [Epoch 11/200] Avg loss: 4.0949
[04:11:55] [Epoch 12/200] Avg loss: 4.0551
[04:11:57] [Epoch 13/200] Avg loss: 3.9899
[04:11:58] [Epoch 14/200] Avg loss: 3.9522
[04:12:00] [Epoch 15/200] Avg loss: 3.8981
[04:12:02] [Epoch 16/200] Avg loss: 3.8568
[04:12:04] [Epoch 17/200] Avg loss: 3.7761
[04:12:05] [Epoch 18/200] Avg loss: 3.7165
[04:12:07] [Epoch 19/200] Avg loss: 3.6572
[04:12:16] [Epoch 20/200] Avg loss: 3.6317 | HR@10: 0.7159 NDCG@10: 0.4653
[04:12:18] [Epoch 21/200] Avg loss: 3.6075
[04:12:19] [Epoch 22/200] Avg loss: 3.5521
[04:12:21] [Epoch 23/200] Avg loss: 3.5372
[04:12:22] [Epoch 24/200] Avg loss: 3.4800
[04:12:24] [Epoch 25/200] Avg loss: 3.4747
[04:12:26] [Epoch 26/200] Avg loss: 3.4110
[04:12:27] [Epoch 27/200] Avg loss: 3.3647
[04:12:29] [Epoch 28/200] Avg loss: 3.3360
[04:12:31] [Epoch 29/200] Avg loss: 3.3124
[04:12:32] [Epoch 30/200] Avg loss: 3.2947
[04:12:34] [Epoch 31/200] Avg loss: 3.2568
[04:12:36] [Epoch 32/200] Avg loss: 3.2322
[04:12:37] [Epoch 33/200] Avg loss: 3.2270
[04:12:39] [Epoch 34/200] Avg loss: 3.1792
[04:12:41] [Epoch 35/200] Avg loss: 3.1907
[04:12:42] [Epoch 36/200] Avg loss: 3.1762
[04:12:44] [Epoch 37/200] Avg loss: 3.1325
[04:12:46] [Epoch 38/200] Avg loss: 3.1363
[04:12:48] [Epoch 39/200] Avg loss: 3.1248
[04:12:57] [Epoch 40/200] Avg loss: 3.1060 | HR@10: 0.8000 NDCG@10: 0.5704
[04:12:58] [Epoch 41/200] Avg loss: 3.0919
[04:13:00] [Epoch 42/200] Avg loss: 3.0852
[04:13:02] [Epoch 43/200] Avg loss: 3.0430
[04:13:03] [Epoch 44/200] Avg loss: 3.0656
[04:13:05] [Epoch 45/200] Avg loss: 3.0397
[04:13:07] [Epoch 46/200] Avg loss: 3.0288
[04:13:08] [Epoch 47/200] Avg loss: 3.0376
[04:13:10] [Epoch 48/200] Avg loss: 2.9918
[04:13:12] [Epoch 49/200] Avg loss: 2.9866
[04:13:13] [Epoch 50/200] Avg loss: 3.0086
[04:13:15] [Epoch 51/200] Avg loss: 2.9958
[04:13:17] [Epoch 52/200] Avg loss: 2.9761
[04:13:18] [Epoch 53/200] Avg loss: 2.9684
[04:13:20] [Epoch 54/200] Avg loss: 2.9499
[04:13:22] [Epoch 55/200] Avg loss: 2.9557
[04:13:23] [Epoch 56/200] Avg loss: 2.9315
[04:13:25] [Epoch 57/200] Avg loss: 2.9517
[04:13:27] [Epoch 58/200] Avg loss: 2.9167
[04:13:28] [Epoch 59/200] Avg loss: 2.9286
[04:13:38] [Epoch 60/200] Avg loss: 2.9207 | HR@10: 0.8238 NDCG@10: 0.6017
[04:13:39] [Epoch 61/200] Avg loss: 2.9210
[04:13:41] [Epoch 62/200] Avg loss: 2.9108
[04:13:43] [Epoch 63/200] Avg loss: 2.8966
[04:13:45] [Epoch 64/200] Avg loss: 2.9088
[04:13:46] [Epoch 65/200] Avg loss: 2.8695
[04:13:48] [Epoch 66/200] Avg loss: 2.8660
[04:13:50] [Epoch 67/200] Avg loss: 2.8854
[04:13:51] [Epoch 68/200] Avg loss: 2.8655
[04:13:53] [Epoch 69/200] Avg loss: 2.8743
[04:13:55] [Epoch 70/200] Avg loss: 2.8525
[04:13:56] [Epoch 71/200] Avg loss: 2.8442
[04:13:58] [Epoch 72/200] Avg loss: 2.8336
[04:14:00] [Epoch 73/200] Avg loss: 2.8326
[04:14:02] [Epoch 74/200] Avg loss: 2.8748
[04:14:03] [Epoch 75/200] Avg loss: 2.8439
[04:14:05] [Epoch 76/200] Avg loss: 2.8425
[04:14:07] [Epoch 77/200] Avg loss: 2.8362
[04:14:08] [Epoch 78/200] Avg loss: 2.8355
[04:14:10] [Epoch 79/200] Avg loss: 2.7983
[04:14:19] [Epoch 80/200] Avg loss: 2.8197 | HR@10: 0.8333 NDCG@10: 0.6188
[04:14:21] [Epoch 81/200] Avg loss: 2.8145
[04:14:23] [Epoch 82/200] Avg loss: 2.8103
[04:14:24] [Epoch 83/200] Avg loss: 2.8116
[04:14:26] [Epoch 84/200] Avg loss: 2.7936
[04:14:28] [Epoch 85/200] Avg loss: 2.7993
[04:14:29] [Epoch 86/200] Avg loss: 2.8135
[04:14:31] [Epoch 87/200] Avg loss: 2.7929
[04:14:33] [Epoch 88/200] Avg loss: 2.7870
[04:14:35] [Epoch 89/200] Avg loss: 2.7814
[04:14:36] [Epoch 90/200] Avg loss: 2.7832
[04:14:38] [Epoch 91/200] Avg loss: 2.7885
[04:14:39] [Epoch 92/200] Avg loss: 2.7511
[04:14:41] [Epoch 93/200] Avg loss: 2.7798
[04:14:43] [Epoch 94/200] Avg loss: 2.7701
[04:14:44] [Epoch 95/200] Avg loss: 2.7517
[04:14:46] [Epoch 96/200] Avg loss: 2.7774
[04:14:48] [Epoch 97/200] Avg loss: 2.7638
[04:14:49] [Epoch 98/200] Avg loss: 2.7487
[04:14:51] [Epoch 99/200] Avg loss: 2.7797
[04:15:00] [Epoch 100/200] Avg loss: 2.7616 | HR@10: 0.8328 NDCG@10: 0.6227
[04:15:02] [Epoch 101/200] Avg loss: 2.7560
[04:15:03] [Epoch 102/200] Avg loss: 2.7317
[04:15:05] [Epoch 103/200] Avg loss: 2.7567
[04:15:07] [Epoch 104/200] Avg loss: 2.7393
[04:15:08] [Epoch 105/200] Avg loss: 2.7337
[04:15:10] [Epoch 106/200] Avg loss: 2.7339
[04:15:11] [Epoch 107/200] Avg loss: 2.7415
[04:15:13] [Epoch 108/200] Avg loss: 2.7553
[04:15:15] [Epoch 109/200] Avg loss: 2.7417
[04:15:16] [Epoch 110/200] Avg loss: 2.7369
[04:15:18] [Epoch 111/200] Avg loss: 2.7173
[04:15:20] [Epoch 112/200] Avg loss: 2.7348
[04:15:21] [Epoch 113/200] Avg loss: 2.7217
[04:15:23] [Epoch 114/200] Avg loss: 2.7308
[04:15:25] [Epoch 115/200] Avg loss: 2.7405
[04:15:26] [Epoch 116/200] Avg loss: 2.7039
[04:15:28] [Epoch 117/200] Avg loss: 2.7304
[04:15:30] [Epoch 118/200] Avg loss: 2.7106
[04:15:31] [Epoch 119/200] Avg loss: 2.7125
[04:15:40] [Epoch 120/200] Avg loss: 2.6962 | HR@10: 0.8351 NDCG@10: 0.6287
[04:15:42] [Epoch 121/200] Avg loss: 2.7200
[04:15:44] [Epoch 122/200] Avg loss: 2.7051
[04:15:45] [Epoch 123/200] Avg loss: 2.6892
[04:15:47] [Epoch 124/200] Avg loss: 2.6854
[04:15:49] [Epoch 125/200] Avg loss: 2.6910
[04:15:50] [Epoch 126/200] Avg loss: 2.7110
[04:15:52] [Epoch 127/200] Avg loss: 2.6880
[04:15:54] [Epoch 128/200] Avg loss: 2.7084
[04:15:55] [Epoch 129/200] Avg loss: 2.7149
[04:15:57] [Epoch 130/200] Avg loss: 2.6853
[04:15:59] [Epoch 131/200] Avg loss: 2.6997
[04:16:00] [Epoch 132/200] Avg loss: 2.6866
[04:16:02] [Epoch 133/200] Avg loss: 2.6950
[04:16:04] [Epoch 134/200] Avg loss: 2.6916
[04:16:05] [Epoch 135/200] Avg loss: 2.6862
[04:16:07] [Epoch 136/200] Avg loss: 2.6883
[04:16:08] [Epoch 137/200] Avg loss: 2.6667
[04:16:10] [Epoch 138/200] Avg loss: 2.6666
[04:16:12] [Epoch 139/200] Avg loss: 2.6824
[04:16:21] [Epoch 140/200] Avg loss: 2.6887 | HR@10: 0.8445 NDCG@10: 0.6369
[04:16:23] [Epoch 141/200] Avg loss: 2.6632
[04:16:24] [Epoch 142/200] Avg loss: 2.6840
[04:16:26] [Epoch 143/200] Avg loss: 2.6712
[04:16:28] [Epoch 144/200] Avg loss: 2.6625
[04:16:29] [Epoch 145/200] Avg loss: 2.6705
[04:16:31] [Epoch 146/200] Avg loss: 2.6566
[04:16:33] [Epoch 147/200] Avg loss: 2.6705
[04:16:34] [Epoch 148/200] Avg loss: 2.6425
[04:16:36] [Epoch 149/200] Avg loss: 2.6824
[04:16:38] [Epoch 150/200] Avg loss: 2.6728
[04:16:39] [Epoch 151/200] Avg loss: 2.6523
[04:16:41] [Epoch 152/200] Avg loss: 2.6613
[04:16:42] [Epoch 153/200] Avg loss: 2.6791
[04:16:44] [Epoch 154/200] Avg loss: 2.6800
[04:16:46] [Epoch 155/200] Avg loss: 2.6524
[04:16:47] [Epoch 156/200] Avg loss: 2.6742
[04:16:49] [Epoch 157/200] Avg loss: 2.6471
[04:16:51] [Epoch 158/200] Avg loss: 2.6779
[04:16:52] [Epoch 159/200] Avg loss: 2.6621
[04:17:02] [Epoch 160/200] Avg loss: 2.6750 | HR@10: 0.8427 NDCG@10: 0.6351
[04:17:03] [Epoch 161/200] Avg loss: 2.6567
[04:17:05] [Epoch 162/200] Avg loss: 2.6699
[04:17:07] [Epoch 163/200] Avg loss: 2.6468
[04:17:08] [Epoch 164/200] Avg loss: 2.6459
[04:17:10] [Epoch 165/200] Avg loss: 2.6596
[04:17:11] [Epoch 166/200] Avg loss: 2.6680
[04:17:13] [Epoch 167/200] Avg loss: 2.6481
[04:17:15] [Epoch 168/200] Avg loss: 2.6372
[04:17:16] [Epoch 169/200] Avg loss: 2.6552
[04:17:18] [Epoch 170/200] Avg loss: 2.6481
[04:17:20] [Epoch 171/200] Avg loss: 2.6321
[04:17:21] [Epoch 172/200] Avg loss: 2.6534
[04:17:23] [Epoch 173/200] Avg loss: 2.6634
[04:17:25] [Epoch 174/200] Avg loss: 2.6294
[04:17:26] [Epoch 175/200] Avg loss: 2.6438
[04:17:28] [Epoch 176/200] Avg loss: 2.6438
[04:17:30] [Epoch 177/200] Avg loss: 2.6488
[04:17:31] [Epoch 178/200] Avg loss: 2.6298
[04:17:33] [Epoch 179/200] Avg loss: 2.6139
[04:17:42] [Epoch 180/200] Avg loss: 2.6373 | HR@10: 0.8455 NDCG@10: 0.6391
[04:17:44] [Epoch 181/200] Avg loss: 2.6403
[04:17:46] [Epoch 182/200] Avg loss: 2.6480
[04:17:47] [Epoch 183/200] Avg loss: 2.6161
[04:17:49] [Epoch 184/200] Avg loss: 2.6497
[04:17:50] [Epoch 185/200] Avg loss: 2.6294
[04:17:52] [Epoch 186/200] Avg loss: 2.6397
[04:17:54] [Epoch 187/200] Avg loss: 2.6320
[04:17:55] [Epoch 188/200] Avg loss: 2.6366
[04:17:57] [Epoch 189/200] Avg loss: 2.6401
[04:17:59] [Epoch 190/200] Avg loss: 2.6337
[04:18:00] [Epoch 191/200] Avg loss: 2.6371
[04:18:02] [Epoch 192/200] Avg loss: 2.6215
[04:18:04] [Epoch 193/200] Avg loss: 2.6228
[04:18:05] [Epoch 194/200] Avg loss: 2.6227
[04:18:07] [Epoch 195/200] Avg loss: 2.6186
[04:18:09] [Epoch 196/200] Avg loss: 2.6056
[04:18:10] [Epoch 197/200] Avg loss: 2.6382
[04:18:12] [Epoch 198/200] Avg loss: 2.6435
[04:18:14] [Epoch 199/200] Avg loss: 2.6458
[04:18:23] [Epoch 200/200] Avg loss: 2.6474 | HR@10: 0.8490 NDCG@10: 0.6392
[04:18:31] [Epochs 200] Test metrics: HR@10: 0.8230 NDCG@10: 0.6225

  Prediction demo (user 1):
    Recent history:  Toy Story (1995), Bug's Life, A (1998), Antz (1998), Hunchback of Notre Dame, The (1996), Hercules (1997)
    Ground truth:    Pocahontas (1995)
    Model top-10:
       1. Lion King, The (1994)
       2. Prince of Egypt, The (1998)
       3. Anastasia (1997)
       4. Pocahontas (1995) <- hit
       5. Rescuers Down Under, The (1990)
       6. Little Mermaid, The (1989)
       7. Babe (1995)
       8. Parent Trap, The (1998)
       9. Fantasia 2000 (1999)
      10. Sword in the Stone, The (1963)
```

### Takeaway

**What changed:** Same sampled softmax loss as v2. Five implementation refinements: truncated normal weight init (`std=1/sqrt(d)`), reversed positional embeddings, registered causal mask buffer, attention weight dropout, and configurable FFN. Model also decomposed into shared `sasrec/` package.

**Results on ML-1M (200 epochs, seed=42):**

| Metric | v2 (SSM) | v4 (SSM + refinements) |
|--------|----------|------------------------|
| Val HR@10 | 0.7922 | 0.8490 |
| Val NDCG@10 | 0.5699 | 0.6392 |
| Test HR@10 | 0.7672 | 0.8230 |
| Test NDCG@10 | 0.5471 | 0.6225 |

**Why it worked:**

- **Weight init** was the biggest contributor. PyTorch defaults produce dot products with variance ~50 (for dim=50), saturating softmax immediately. Truncated normal with `std=1/sqrt(d)` keeps initial scores near zero. v2 started at loss 18.3; v4 started at 5.4 and reached v2's final HR by epoch 40.
- **Reversed positions** give the prediction position a consistent embedding (always 0) instead of varying by sequence length. The model learns "how far back" rather than "what absolute slot." (Huang 2025, arxiv: 2504.09596)
- **Attention dropout** was missing in v1-v3. The original Transformer paper applies dropout on attention weights after softmax, not just on the output. One line, standard practice.

**The lesson:** v1-v3 proved the loss function matters. v4 proves implementation details matter just as much — +7% HR and +14% NDCG from changes that didn't touch the loss or model capacity. D'Ercoli et al. 2025 documented 2-5% swings from these details individually; they compound.

## v5: Scaling to ML-20M - Depth Doesn't Pay

**What changed:** No new code. Ran `train_v4.py` on ML-20M (138K users, 18K items) at 2, 4, and 6 attention blocks to measure whether depth improves metrics on a larger dataset. Shell script `train_v5.sh` automates the experiment. See it for `run` commands used.

### Takeaway

**Results on ML-20M (50 epochs, dim=64, batch_size=256, seed=42):**

| Blocks | Params | Test HR@10 | Test NDCG@10 | Time/epoch |
|--------|--------|------------|--------------|------------|
| 2 | 1,236,928 | 0.9512 | 0.7798 | ~45s |
| 4 | 1,286,848 | 0.9526 | 0.7850 | ~75s |
| 6 | 1,336,768 | 0.9541 | 0.7883 | ~110s |

**Depth doesn't collapse — but it doesn't pay either.** Going from 2 to 6 blocks adds 8% more parameters, takes 2.5x longer per epoch, and gains only +0.3% HR and +1.1% NDCG. Tripling your compute budget for +1% is a bad tradeoff. This is why production SASRec stayed at 2 blocks for years.

**Why no outright collapse?** The papers that documented depth collapse (Peng et al. 2022, Guo et al. 2024 Table 8) used the original SASRec implementation with Post-LN normalization and PyTorch default initialization. Our v4 refinements (Pre-LN, truncated normal init, attention dropout) already mitigate the worst over-smoothing. We accidentally fixed the easy part of the depth problem in v4. The harder part — making depth actually *worth the cost* — requires the architectural changes in v6-v8.

**The over-smoothing mechanism is still present, just delayed.** With more blocks, item representations in the sequence converge toward each other as each attention layer blends them further. At 2 blocks the useful signal is "recent items predict next item" and "similar genres cluster." Blocks 3-6 have diminishing new structure to extract from flat item sequences (unlike language, which has deep syntactic hierarchy for deeper layers to decompose). The representations don't collapse into uniformity, but they stop improving meaningfully.

## v6: ReZero Residual Scaling

**What changed:** Added learnable residual scaling (ReZero) to each attention block. Two scalar parameters per block (one for attention, one for FFN), initialized to zero. This makes each block start as an identity function and gradually learn how much to contribute to the residual stream. 12 extra parameters for 6 blocks.

**Results on ML-20M (6 blocks, dim=64, 50 epochs, seed=42):**

| Config | Test HR@10 | Test NDCG@10 | Final Loss |
|--------|------------|--------------|------------|
| v5: 2 blocks, no scaling | 0.9512 | 0.7798 | 1.7156 |
| v5: 6 blocks, no scaling | 0.9541 | 0.7883 | 1.6194 |
| v6: 6 blocks, ReZero | 0.9553 | 0.7917 | 1.5992 |

**Why ReZero helps (a little).** Without scaling, each block dumps an equal-weight contribution onto the residual stream. After 6 blocks the original embedding is ~14% of the final representation — diluted by accumulated block outputs. ReZero initializes each block as identity (alpha=0), so the model first learns good embeddings, then gradually lets each block contribute what it's learned. The residual stream stays dominated by useful signal.

**Why the gains are modest.** ReZero keeps the residual stream healthy, but deeper blocks still have nothing fundamentally new to learn from flat item sequences. The attention mechanism (softmax over Q·K) saturates at 2-3 blocks worth of sequential patterns. Making depth stable is necessary but not sufficient — deeper blocks also need new information to work with.
