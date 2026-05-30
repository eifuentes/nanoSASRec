"""dataset.py - data loading and batch sampling for nanoSASRec v4+"""

import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np


def load_data(
    data_dir: str, dataset: str
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, List[int]], int, int, Dict[int, str]]:
    """Load preprocessed dataset split from disk.

    Returns: user_train, user_valid, user_test, num_users, num_items, item_id_map
    """
    dataset_split_filepath = f"{data_dir}/{dataset}/split.json"
    assert os.path.exists(dataset_split_filepath), f"Run `python data.py --dataset {dataset}` first"
    with open(dataset_split_filepath) as f:
        dataset_split = json.load(f)

    user_train = {int(k): v for k, v in dataset_split["user_train"].items()}
    user_valid = {int(k): v for k, v in dataset_split["user_valid"].items()}
    user_test = {int(k): v for k, v in dataset_split["user_test"].items()}

    num_users: int = dataset_split["num_users"]
    num_items: int = dataset_split["num_items"]

    item_id_map = {int(k): v for k, v in dataset_split["item_id_map"].items()}

    print(f"Number Users: {num_users:,}\nNumber Items: {num_items:,}")
    return user_train, user_valid, user_test, num_users, num_items, item_id_map


def load_item_titles(data_dir: str, dataset: str, item_id_map: Dict[int, str]) -> Dict[int, str]:
    titles_path = f"{data_dir}/{dataset}/titles.json"
    if not os.path.exists(titles_path):
        return {}
    with open(titles_path) as f:
        raw_titles = json.load(f)
    return {idx: raw_titles[raw_id] for idx, raw_id in item_id_map.items() if raw_id in raw_titles}


def sample_batch(
    user_train: Dict[int, List[int]], num_users: int, num_items: int, max_len: int, batch_size: int, num_negs: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs a training batch for autoregressive sequence recommendation models like SASRec.

    Returns three numpy arrays each comprised of sampled users of batch_size to create the batch
        - input of user's sequences truncated to remove the last item, shape(batch_size, max_len -1)
        - positive targets are the shifted-by-one ground truth next items, shape(batch_size, max_len -1)
        - target of per user unseen items treated as negatives to be used in contrastive training, shape(batch_size, num_negs)

    We use NumPy arrays for the batches because they are
    memory-contiguous, which is required for efficient GPU processing.
    PyTorch later handles these NumPy arrays very gracefully.
    """
    # Initialize the modern NumPy random generator (faster than np.random.randint)
    rng = np.random.default_rng()

    # 1. Sample users
    # NOTE: random.sample is 'without replacement', gives unique users
    batch_users = random.sample(range(1, num_users + 1), k=batch_size)

    # 2. Pre-allocate matrices in memory of type int64 which is equal to torch LongTensor later
    # NOTE: (max_len - 1) is a result of the position prediction setup of autoregressive models like SASRec
    input_seq_batch = np.zeros((batch_size, max_len - 1), dtype=np.int64)
    target_pos_seq_batch = np.zeros((batch_size, max_len - 1), dtype=np.int64)
    # NOTE: We sample a set of negatives per user that get reused for each position/target combination
    target_neg_seq_batch = np.zeros((batch_size, num_negs), dtype=np.int64)

    # each record in the batch is a user sampled
    for i, user_id in enumerate(batch_users):
        user_seq: List[int] = user_train[user_id]

        # Convert to set once to be more performant
        # in python 'item in list' is O(N) -> slow. 'item in set' is O(1) -> instant.
        # we check against the full user history before truncation so negatives are truly "unseen" items.
        seen_items = set(user_seq)

        # 3. Handle sequence padding or truncation depending on length of sequence
        user_seq_len = len(user_seq)
        if user_seq_len >= max_len:
            # Keep only the 'max_len' most recent interactions
            # NOTE: the sequence is ordered by time starting with oldest to newest
            user_seq = user_seq[user_seq_len - max_len :]
        else:
            # Pad with 0 at the beginning (pre-padding)
            user_seq = [0] * (max_len - user_seq_len) + user_seq

        # 4. Shift sequences for autoregressive prediction
        # input is 0 to t-1, target is 1 to t
        input_user_seq = user_seq[:-1]
        target_user_seq = user_seq[1:]

        # 5. Sample `num_negs` negatives per user using a rejection sampling approach
        user_negs = np.zeros(num_negs, dtype=np.int64)

        # Rejection sampling loop
        count = 0
        while count < num_negs:
            # Sample a block of random IDs in C-speed, then filter in python.
            # NOTE: `num_items + 1` is due to exclusive nature of rng.integers on the high side so all item ids are included
            # NOTE: we start with 1 in rng.integers since 0 is reserved for the padding token in the item ID lookups
            samples = rng.integers(1, num_items + 1, size=num_negs - count)
            for s in samples:
                if s not in seen_items:
                    user_negs[count] = s
                    count += 1
                    if count == num_negs:
                        break

        # 6. Move local sequences into the batch matrices
        input_seq_batch[i, :] = input_user_seq
        target_pos_seq_batch[i, :] = target_user_seq
        target_neg_seq_batch[i, :] = user_negs

    return input_seq_batch, target_pos_seq_batch, target_neg_seq_batch
