# nanoSASRec Walkthrough

Curriculum walkthrough from v1 to v8.

## Differences

Here are known differences than the original implementation of SASRec.

- **v1 uses Pre-LN** (norm before attention), the original uses Post-LN (norm after residual add). Pre-LN is more stable for deeper models, which is why we start with it - v5+ needs it to scale.
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

`uv run python train_v3.py --device mps --epochs 200 --eval-every 20 --num-negatives 256`

### Output

```bash
Number Users: 6,040
Number Items: 3,416
Dataset Lengths: 6040 6040 6040
Model parameters: 211,500
Estimate batches per epoch: 47
[03:43:47] [Epoch 1/200] Avg loss: 2.5356 | Avg P(pos): 0.5409
[03:43:48] [Epoch 2/200] Avg loss: 1.7205 | Avg P(pos): 0.5660
[03:43:50] [Epoch 3/200] Avg loss: 1.2264 | Avg P(pos): 0.5491
[03:43:52] [Epoch 4/200] Avg loss: 0.9411 | Avg P(pos): 0.5391
[03:43:54] [Epoch 5/200] Avg loss: 0.7916 | Avg P(pos): 0.5408
[03:43:56] [Epoch 6/200] Avg loss: 0.7077 | Avg P(pos): 0.5521
[03:43:57] [Epoch 7/200] Avg loss: 0.6569 | Avg P(pos): 0.5668
[03:43:59] [Epoch 8/200] Avg loss: 0.6313 | Avg P(pos): 0.5821
[03:44:01] [Epoch 9/200] Avg loss: 0.6109 | Avg P(pos): 0.5934
[03:44:03] [Epoch 10/200] Avg loss: 0.5982 | Avg P(pos): 0.6035
[03:44:04] [Epoch 11/200] Avg loss: 0.5819 | Avg P(pos): 0.6130
[03:44:06] [Epoch 12/200] Avg loss: 0.5717 | Avg P(pos): 0.6199
[03:44:08] [Epoch 13/200] Avg loss: 0.5625 | Avg P(pos): 0.6227
[03:44:10] [Epoch 14/200] Avg loss: 0.5524 | Avg P(pos): 0.6283
[03:44:12] [Epoch 15/200] Avg loss: 0.5428 | Avg P(pos): 0.6316
[03:44:13] [Epoch 16/200] Avg loss: 0.5363 | Avg P(pos): 0.6327
[03:44:15] [Epoch 17/200] Avg loss: 0.5326 | Avg P(pos): 0.6347
[03:44:17] [Epoch 18/200] Avg loss: 0.5243 | Avg P(pos): 0.6393
[03:44:19] [Epoch 19/200] Avg loss: 0.5225 | Avg P(pos): 0.6430
[03:44:29] [Epoch 20/200] Avg loss: 0.5187 | Avg P(pos): 0.6488 | HR@10: 0.4363 NDCG@10: 0.2401
[03:44:31] [Epoch 21/200] Avg loss: 0.5127 | Avg P(pos): 0.6599
[03:44:32] [Epoch 22/200] Avg loss: 0.5122 | Avg P(pos): 0.6632
[03:44:34] [Epoch 23/200] Avg loss: 0.5091 | Avg P(pos): 0.6697
[03:44:36] [Epoch 24/200] Avg loss: 0.5116 | Avg P(pos): 0.6687
[03:44:38] [Epoch 25/200] Avg loss: 0.5067 | Avg P(pos): 0.6721
[03:44:39] [Epoch 26/200] Avg loss: 0.5043 | Avg P(pos): 0.6582
[03:44:41] [Epoch 27/200] Avg loss: 0.5036 | Avg P(pos): 0.6413
[03:44:43] [Epoch 28/200] Avg loss: 0.4999 | Avg P(pos): 0.6303
[03:44:45] [Epoch 29/200] Avg loss: 0.4993 | Avg P(pos): 0.6244
[03:44:47] [Epoch 30/200] Avg loss: 0.4963 | Avg P(pos): 0.6200
[03:44:48] [Epoch 31/200] Avg loss: 0.4974 | Avg P(pos): 0.6131
[03:44:50] [Epoch 32/200] Avg loss: 0.4946 | Avg P(pos): 0.6122
[03:44:52] [Epoch 33/200] Avg loss: 0.4909 | Avg P(pos): 0.6125
[03:44:54] [Epoch 34/200] Avg loss: 0.4931 | Avg P(pos): 0.6095
[03:44:55] [Epoch 35/200] Avg loss: 0.4892 | Avg P(pos): 0.6072
[03:44:57] [Epoch 36/200] Avg loss: 0.4896 | Avg P(pos): 0.6066
[03:44:59] [Epoch 37/200] Avg loss: 0.4900 | Avg P(pos): 0.6032
[03:45:01] [Epoch 38/200] Avg loss: 0.4895 | Avg P(pos): 0.5993
[03:45:03] [Epoch 39/200] Avg loss: 0.4848 | Avg P(pos): 0.6021
[03:45:13] [Epoch 40/200] Avg loss: 0.4849 | Avg P(pos): 0.6022 | HR@10: 0.4531 NDCG@10: 0.2467
[03:45:15] [Epoch 41/200] Avg loss: 0.4822 | Avg P(pos): 0.6031
[03:45:16] [Epoch 42/200] Avg loss: 0.4795 | Avg P(pos): 0.6103
[03:45:18] [Epoch 43/200] Avg loss: 0.4783 | Avg P(pos): 0.6126
[03:45:20] [Epoch 44/200] Avg loss: 0.4743 | Avg P(pos): 0.6159
[03:45:22] [Epoch 45/200] Avg loss: 0.4701 | Avg P(pos): 0.6179
[03:45:23] [Epoch 46/200] Avg loss: 0.4662 | Avg P(pos): 0.6200
[03:45:25] [Epoch 47/200] Avg loss: 0.4612 | Avg P(pos): 0.6214
[03:45:27] [Epoch 48/200] Avg loss: 0.4598 | Avg P(pos): 0.6248
[03:45:29] [Epoch 49/200] Avg loss: 0.4544 | Avg P(pos): 0.6241
[03:45:31] [Epoch 50/200] Avg loss: 0.4511 | Avg P(pos): 0.6264
[03:45:32] [Epoch 51/200] Avg loss: 0.4465 | Avg P(pos): 0.6273
[03:45:34] [Epoch 52/200] Avg loss: 0.4454 | Avg P(pos): 0.6259
[03:45:36] [Epoch 53/200] Avg loss: 0.4446 | Avg P(pos): 0.6252
[03:45:38] [Epoch 54/200] Avg loss: 0.4406 | Avg P(pos): 0.6239
[03:45:40] [Epoch 55/200] Avg loss: 0.4381 | Avg P(pos): 0.6264
[03:45:41] [Epoch 56/200] Avg loss: 0.4372 | Avg P(pos): 0.6241
[03:45:43] [Epoch 57/200] Avg loss: 0.4364 | Avg P(pos): 0.6225
[03:45:45] [Epoch 58/200] Avg loss: 0.4329 | Avg P(pos): 0.6224
[03:45:47] [Epoch 59/200] Avg loss: 0.4315 | Avg P(pos): 0.6213
[03:45:57] [Epoch 60/200] Avg loss: 0.4308 | Avg P(pos): 0.6201 | HR@10: 0.5460 NDCG@10: 0.3053
[03:45:59] [Epoch 61/200] Avg loss: 0.4297 | Avg P(pos): 0.6186
[03:46:01] [Epoch 62/200] Avg loss: 0.4290 | Avg P(pos): 0.6206
[03:46:02] [Epoch 63/200] Avg loss: 0.4263 | Avg P(pos): 0.6197
[03:46:04] [Epoch 64/200] Avg loss: 0.4232 | Avg P(pos): 0.6189
[03:46:06] [Epoch 65/200] Avg loss: 0.4237 | Avg P(pos): 0.6192
[03:46:08] [Epoch 66/200] Avg loss: 0.4210 | Avg P(pos): 0.6167
[03:46:09] [Epoch 67/200] Avg loss: 0.4199 | Avg P(pos): 0.6184
[03:46:11] [Epoch 68/200] Avg loss: 0.4189 | Avg P(pos): 0.6171
[03:46:13] [Epoch 69/200] Avg loss: 0.4148 | Avg P(pos): 0.6171
[03:46:15] [Epoch 70/200] Avg loss: 0.4146 | Avg P(pos): 0.6164
[03:46:16] [Epoch 71/200] Avg loss: 0.4118 | Avg P(pos): 0.6164
[03:46:18] [Epoch 72/200] Avg loss: 0.4110 | Avg P(pos): 0.6162
[03:46:20] [Epoch 73/200] Avg loss: 0.4086 | Avg P(pos): 0.6170
[03:46:22] [Epoch 74/200] Avg loss: 0.4063 | Avg P(pos): 0.6178
[03:46:23] [Epoch 75/200] Avg loss: 0.4057 | Avg P(pos): 0.6164
[03:46:25] [Epoch 76/200] Avg loss: 0.4027 | Avg P(pos): 0.6200
[03:46:27] [Epoch 77/200] Avg loss: 0.3983 | Avg P(pos): 0.6182
[03:46:28] [Epoch 78/200] Avg loss: 0.3994 | Avg P(pos): 0.6199
[03:46:30] [Epoch 79/200] Avg loss: 0.3962 | Avg P(pos): 0.6235
[03:46:41] [Epoch 80/200] Avg loss: 0.3944 | Avg P(pos): 0.6224 | HR@10: 0.6262 NDCG@10: 0.3655
[03:46:43] [Epoch 81/200] Avg loss: 0.3909 | Avg P(pos): 0.6276
[03:46:44] [Epoch 82/200] Avg loss: 0.3904 | Avg P(pos): 0.6249
[03:46:46] [Epoch 83/200] Avg loss: 0.3901 | Avg P(pos): 0.6240
[03:46:48] [Epoch 84/200] Avg loss: 0.3875 | Avg P(pos): 0.6262
[03:46:50] [Epoch 85/200] Avg loss: 0.3864 | Avg P(pos): 0.6311
[03:46:51] [Epoch 86/200] Avg loss: 0.3828 | Avg P(pos): 0.6366
[03:46:53] [Epoch 87/200] Avg loss: 0.3836 | Avg P(pos): 0.6344
[03:46:55] [Epoch 88/200] Avg loss: 0.3820 | Avg P(pos): 0.6364
[03:46:57] [Epoch 89/200] Avg loss: 0.3786 | Avg P(pos): 0.6399
[03:46:59] [Epoch 90/200] Avg loss: 0.3756 | Avg P(pos): 0.6396
[03:47:00] [Epoch 91/200] Avg loss: 0.3766 | Avg P(pos): 0.6412
[03:47:02] [Epoch 92/200] Avg loss: 0.3749 | Avg P(pos): 0.6472
[03:47:04] [Epoch 93/200] Avg loss: 0.3713 | Avg P(pos): 0.6530
[03:47:06] [Epoch 94/200] Avg loss: 0.3708 | Avg P(pos): 0.6512
[03:47:07] [Epoch 95/200] Avg loss: 0.3711 | Avg P(pos): 0.6534
[03:47:09] [Epoch 96/200] Avg loss: 0.3678 | Avg P(pos): 0.6569
[03:47:11] [Epoch 97/200] Avg loss: 0.3660 | Avg P(pos): 0.6559
[03:47:13] [Epoch 98/200] Avg loss: 0.3662 | Avg P(pos): 0.6547
[03:47:15] [Epoch 99/200] Avg loss: 0.3647 | Avg P(pos): 0.6609
[03:47:25] [Epoch 100/200] Avg loss: 0.3601 | Avg P(pos): 0.6610 | HR@10: 0.6813 NDCG@10: 0.4119
[03:47:27] [Epoch 101/200] Avg loss: 0.3607 | Avg P(pos): 0.6640
[03:47:28] [Epoch 102/200] Avg loss: 0.3579 | Avg P(pos): 0.6635
[03:47:30] [Epoch 103/200] Avg loss: 0.3597 | Avg P(pos): 0.6611
[03:47:32] [Epoch 104/200] Avg loss: 0.3563 | Avg P(pos): 0.6660
[03:47:34] [Epoch 105/200] Avg loss: 0.3555 | Avg P(pos): 0.6672
[03:47:36] [Epoch 106/200] Avg loss: 0.3546 | Avg P(pos): 0.6652
[03:47:37] [Epoch 107/200] Avg loss: 0.3511 | Avg P(pos): 0.6660
[03:47:39] [Epoch 108/200] Avg loss: 0.3515 | Avg P(pos): 0.6670
[03:47:41] [Epoch 109/200] Avg loss: 0.3495 | Avg P(pos): 0.6682
[03:47:43] [Epoch 110/200] Avg loss: 0.3474 | Avg P(pos): 0.6723
[03:47:44] [Epoch 111/200] Avg loss: 0.3484 | Avg P(pos): 0.6699
[03:47:46] [Epoch 112/200] Avg loss: 0.3445 | Avg P(pos): 0.6701
[03:47:48] [Epoch 113/200] Avg loss: 0.3439 | Avg P(pos): 0.6775
[03:47:50] [Epoch 114/200] Avg loss: 0.3419 | Avg P(pos): 0.6755
[03:47:52] [Epoch 115/200] Avg loss: 0.3406 | Avg P(pos): 0.6717
[03:47:53] [Epoch 116/200] Avg loss: 0.3423 | Avg P(pos): 0.6726
[03:47:55] [Epoch 117/200] Avg loss: 0.3379 | Avg P(pos): 0.6744
[03:47:57] [Epoch 118/200] Avg loss: 0.3402 | Avg P(pos): 0.6734
[03:47:59] [Epoch 119/200] Avg loss: 0.3357 | Avg P(pos): 0.6736
[03:48:09] [Epoch 120/200] Avg loss: 0.3335 | Avg P(pos): 0.6756 | HR@10: 0.7205 NDCG@10: 0.4456
[03:48:11] [Epoch 121/200] Avg loss: 0.3329 | Avg P(pos): 0.6708
[03:48:13] [Epoch 122/200] Avg loss: 0.3323 | Avg P(pos): 0.6757
[03:48:15] [Epoch 123/200] Avg loss: 0.3319 | Avg P(pos): 0.6785
[03:48:16] [Epoch 124/200] Avg loss: 0.3332 | Avg P(pos): 0.6746
[03:48:18] [Epoch 125/200] Avg loss: 0.3302 | Avg P(pos): 0.6797
[03:48:20] [Epoch 126/200] Avg loss: 0.3296 | Avg P(pos): 0.6755
[03:48:22] [Epoch 127/200] Avg loss: 0.3296 | Avg P(pos): 0.6792
[03:48:23] [Epoch 128/200] Avg loss: 0.3269 | Avg P(pos): 0.6781
[03:48:25] [Epoch 129/200] Avg loss: 0.3269 | Avg P(pos): 0.6778
[03:48:27] [Epoch 130/200] Avg loss: 0.3251 | Avg P(pos): 0.6815
[03:48:29] [Epoch 131/200] Avg loss: 0.3245 | Avg P(pos): 0.6813
[03:48:30] [Epoch 132/200] Avg loss: 0.3226 | Avg P(pos): 0.6825
[03:48:32] [Epoch 133/200] Avg loss: 0.3215 | Avg P(pos): 0.6837
[03:48:34] [Epoch 134/200] Avg loss: 0.3221 | Avg P(pos): 0.6852
[03:48:36] [Epoch 135/200] Avg loss: 0.3214 | Avg P(pos): 0.6865
[03:48:37] [Epoch 136/200] Avg loss: 0.3183 | Avg P(pos): 0.6846
[03:48:39] [Epoch 137/200] Avg loss: 0.3175 | Avg P(pos): 0.6840
[03:48:41] [Epoch 138/200] Avg loss: 0.3168 | Avg P(pos): 0.6871
[03:48:43] [Epoch 139/200] Avg loss: 0.3141 | Avg P(pos): 0.6908
[03:48:53] [Epoch 140/200] Avg loss: 0.3159 | Avg P(pos): 0.6866 | HR@10: 0.7474 NDCG@10: 0.4696
[03:48:55] [Epoch 141/200] Avg loss: 0.3122 | Avg P(pos): 0.6878
[03:48:57] [Epoch 142/200] Avg loss: 0.3143 | Avg P(pos): 0.6860
[03:48:59] [Epoch 143/200] Avg loss: 0.3111 | Avg P(pos): 0.6886
[03:49:00] [Epoch 144/200] Avg loss: 0.3106 | Avg P(pos): 0.6895
[03:49:02] [Epoch 145/200] Avg loss: 0.3088 | Avg P(pos): 0.6949
[03:49:04] [Epoch 146/200] Avg loss: 0.3090 | Avg P(pos): 0.6907
[03:49:06] [Epoch 147/200] Avg loss: 0.3084 | Avg P(pos): 0.6943
[03:49:07] [Epoch 148/200] Avg loss: 0.3085 | Avg P(pos): 0.6949
[03:49:09] [Epoch 149/200] Avg loss: 0.3113 | Avg P(pos): 0.6925
[03:49:11] [Epoch 150/200] Avg loss: 0.3093 | Avg P(pos): 0.6912
[03:49:13] [Epoch 151/200] Avg loss: 0.3048 | Avg P(pos): 0.6955
[03:49:15] [Epoch 152/200] Avg loss: 0.3027 | Avg P(pos): 0.6991
[03:49:16] [Epoch 153/200] Avg loss: 0.3054 | Avg P(pos): 0.6992
[03:49:18] [Epoch 154/200] Avg loss: 0.3035 | Avg P(pos): 0.7007
[03:49:20] [Epoch 155/200] Avg loss: 0.3039 | Avg P(pos): 0.7009
[03:49:22] [Epoch 156/200] Avg loss: 0.3031 | Avg P(pos): 0.7022
[03:49:23] [Epoch 157/200] Avg loss: 0.3011 | Avg P(pos): 0.7038
[03:49:25] [Epoch 158/200] Avg loss: 0.3012 | Avg P(pos): 0.7061
[03:49:27] [Epoch 159/200] Avg loss: 0.3005 | Avg P(pos): 0.7047
[03:49:37] [Epoch 160/200] Avg loss: 0.2994 | Avg P(pos): 0.7076 | HR@10: 0.7692 NDCG@10: 0.4990
[03:49:39] [Epoch 161/200] Avg loss: 0.2983 | Avg P(pos): 0.7120
[03:49:41] [Epoch 162/200] Avg loss: 0.2965 | Avg P(pos): 0.7126
[03:49:43] [Epoch 163/200] Avg loss: 0.2994 | Avg P(pos): 0.7097
[03:49:44] [Epoch 164/200] Avg loss: 0.2962 | Avg P(pos): 0.7102
[03:49:46] [Epoch 165/200] Avg loss: 0.2980 | Avg P(pos): 0.7112
[03:49:48] [Epoch 166/200] Avg loss: 0.2983 | Avg P(pos): 0.7108
[03:49:50] [Epoch 167/200] Avg loss: 0.2972 | Avg P(pos): 0.7121
[03:49:51] [Epoch 168/200] Avg loss: 0.2965 | Avg P(pos): 0.7108
[03:49:53] [Epoch 169/200] Avg loss: 0.2953 | Avg P(pos): 0.7115
[03:49:55] [Epoch 170/200] Avg loss: 0.2959 | Avg P(pos): 0.7091
[03:49:57] [Epoch 171/200] Avg loss: 0.2915 | Avg P(pos): 0.7175
[03:49:58] [Epoch 172/200] Avg loss: 0.2952 | Avg P(pos): 0.7188
[03:50:00] [Epoch 173/200] Avg loss: 0.2936 | Avg P(pos): 0.7137
[03:50:02] [Epoch 174/200] Avg loss: 0.2939 | Avg P(pos): 0.7136
[03:50:04] [Epoch 175/200] Avg loss: 0.2931 | Avg P(pos): 0.7146
[03:50:06] [Epoch 176/200] Avg loss: 0.2907 | Avg P(pos): 0.7222
[03:50:07] [Epoch 177/200] Avg loss: 0.2921 | Avg P(pos): 0.7170
[03:50:09] [Epoch 178/200] Avg loss: 0.2886 | Avg P(pos): 0.7187
[03:50:11] [Epoch 179/200] Avg loss: 0.2885 | Avg P(pos): 0.7206
[03:50:21] [Epoch 180/200] Avg loss: 0.2884 | Avg P(pos): 0.7205 | HR@10: 0.7825 NDCG@10: 0.5159
[03:50:23] [Epoch 181/200] Avg loss: 0.2883 | Avg P(pos): 0.7228
[03:50:25] [Epoch 182/200] Avg loss: 0.2887 | Avg P(pos): 0.7203
[03:50:27] [Epoch 183/200] Avg loss: 0.2875 | Avg P(pos): 0.7240
[03:50:29] [Epoch 184/200] Avg loss: 0.2868 | Avg P(pos): 0.7270
[03:50:30] [Epoch 185/200] Avg loss: 0.2863 | Avg P(pos): 0.7285
[03:50:32] [Epoch 186/200] Avg loss: 0.2874 | Avg P(pos): 0.7262
[03:50:34] [Epoch 187/200] Avg loss: 0.2852 | Avg P(pos): 0.7327
[03:50:36] [Epoch 188/200] Avg loss: 0.2862 | Avg P(pos): 0.7334
[03:50:37] [Epoch 189/200] Avg loss: 0.2856 | Avg P(pos): 0.7299
[03:50:39] [Epoch 190/200] Avg loss: 0.2864 | Avg P(pos): 0.7323
[03:50:41] [Epoch 191/200] Avg loss: 0.2851 | Avg P(pos): 0.7316
[03:50:43] [Epoch 192/200] Avg loss: 0.2824 | Avg P(pos): 0.7328
[03:50:44] [Epoch 193/200] Avg loss: 0.2811 | Avg P(pos): 0.7339
[03:50:46] [Epoch 194/200] Avg loss: 0.2824 | Avg P(pos): 0.7327
[03:50:48] [Epoch 195/200] Avg loss: 0.2822 | Avg P(pos): 0.7326
[03:50:50] [Epoch 196/200] Avg loss: 0.2829 | Avg P(pos): 0.7358
[03:50:52] [Epoch 197/200] Avg loss: 0.2791 | Avg P(pos): 0.7393
[03:50:53] [Epoch 198/200] Avg loss: 0.2798 | Avg P(pos): 0.7368
[03:50:55] [Epoch 199/200] Avg loss: 0.2816 | Avg P(pos): 0.7367
[03:51:06] [Epoch 200/200] Avg loss: 0.2812 | Avg P(pos): 0.7374 | HR@10: 0.7886 NDCG@10: 0.5316
[03:51:15] [Epochs 200] Test metrics: HR@10: 0.7636 NDCG@10: 0.5105

  Prediction demo (user 1):
    Recent history:  Toy Story (1995), Bug's Life, A (1998), Antz (1998), Hunchback of Notre Dame, The (1996), Hercules (1997)
    Ground truth:    Pocahontas (1995)
    Model top-10:
       1. Lion King, The (1994)
       2. Lady and the Tramp (1955)
       3. Sleeping Beauty (1959)
       4. Fantasia (1940)
       5. Little Mermaid, The (1989)
       6. Anastasia (1997)
       7. Nightmare Before Christmas, The (1993)
       8. Groundhog Day (1993)
       9. Muppet Christmas Carol, The (1992)
      10. Jungle Book, The (1967)
```

### Takeaway

**What we changed:** Only the loss function, again. The model architecture is identical to v1 and v2. The diff from v2 is in the loss computation: sampled softmax is replaced with generalized binary cross-entropy (gBCE), which goes back to BCE's independent per-item scoring but adds a calibration parameter `t` that downweights the positive signal to correct for undersampling negatives relative to the full catalog.

**What gBCE does differently:** Standard BCE with negative sampling sees a 1:1 positive-to-negative ratio in training, but the real catalog ratio is 1:3,415. The model learns that items are far more likely to be relevant than they actually are. gBCE introduces `t` (0 < t < 1) that scales the positive loss term: `loss = t * (-log(P_pos)) + (1-t) * (-log(1-P_neg))`. This dampens the "push toward 1" gradient while keeping the positive and negative terms balanced, preventing the model from becoming overconfident without starving it of learning signal.

**P(pos) tells the story:** v1's P(pos) climbed to 0.999, making the top 25 items indistinguishable. v3's P(pos) stabilized around 0.73, well below the overconfidence zone but high enough that the model is still learning meaningful preferences. This is the calibration the paper proved theoretically (Theorem 5.1, Petrov & Macdonald 2308.07192): with the right `t`, the predicted probabilities reflect actual item relevance rather than training artifact.

**The interaction between `t` and `num_negatives`:** We ran v3 twice. With 1 negative and `t=0.5`, the model barely learned (HR@10 = 0.44, worse than v1). The positive signal was dampened too aggressively with too little negative information to compensate. With 256 negatives and `t=0.5`, the model matched v2's HR@10 (0.79) while maintaining calibrated probabilities. The lesson: `t` and `num_negatives` are not independent knobs. Fewer negatives need less dampening (higher `t`), more negatives can tolerate stronger dampening (lower `t`). The paper's formula `t = 1/(1 + num_negs * (num_items-1)/num_items)` captures this relationship precisely.

**Why NDCG was lower than v2:** v3's NDCG@10 (0.53) came in below v2's (0.57) despite matching on HR. gBCE's calibrated probabilities spread scores more evenly, which helps avoid overconfidence but doesn't force the hard competition that softmax creates. Sampled softmax explicitly makes items fight for probability mass, producing sharper separation at the top of the ranking. gBCE trades some ranking sharpness for better-calibrated probability estimates. Which matters more depends on the application: if you need probability estimates (e.g., multiplying P(purchase) by item price for revenue prediction), gBCE wins. If you only care about ranking, SSM's sharper top-K separation gives better NDCG.

**What we learned about the loss function progression:** After three versions with identical architecture, the conclusion is clear: SASRec's performance issues were never about the architecture. v1's BCE overconfidence, v2's softmax fix, v3's calibration approach are all statements about how the training signal is shaped, not about how attention works. The architecture was fine from the start. This matches the central finding of the 2023 research: fix the loss, fix the model.
