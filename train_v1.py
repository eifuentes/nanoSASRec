"""
train_v1.py
"""
import argparse
import json
from typing import Dict, Tuple, List
import random

import numpy as np

DATASETS = ("ml-1m", "ml-20m", "beauty")

def load_data(data_dir: str, dataset: str) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, List[int]], int, int, Dict[int, str]]:
    """Load preprocessed dataset split from disk.

    Returns: user_train, user_valid, user_test, num_users, num_items, item_id_map
    """
    dataset_split_filepath = f"{data_dir}/{dataset}/split.json"
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

def sample_batch(user_train: Dict[int, List[int]], num_users: int, num_items: int, maxlen: int, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Select random users of size batch_size
    batch_users = random.sample(range(1, num_users + 1), k=batch_size)
    # allocate input, target, and negative sequence batch matrices
    input_seq_batch = np.zeros((batch_size, maxlen - 1), int)
    target_pos_seq_batch = np.zeros((batch_size, maxlen - 1), int)
    target_neg_seq_batch = np.zeros((batch_size, maxlen - 1), int)
    # extract user sequence and truncate or pad sequence to maxlen
    for i, user in enumerate(batch_users):
        user_seq = user_train[user]
        seqlen = len(user_seq)
        if seqlen >= maxlen:
            # User sequences are temporally ordered from oldest to newest
            # For these models its best practice to keep the most recent interactions
            start_idx = seqlen - maxlen
            user_seq = user_seq[start_idx:]
        else:
            # 0 is reserved value for padding values
            # TODO: given np array is init with 0s this is redudant, could insert at correct index
            num_pads = maxlen - seqlen
            pad_seq = [0] * num_pads
            user_seq = pad_seq + user_seq
        # create input and target sequences
        # create additional sub-sequences from a single sequence
        # ideally done by shifting the input sequence left and lining up
        input_seq, target_seq = user_seq[:-1], user_seq[1:]  # each seq is maxlen - 1
        # select random negatives from unseen items throuhgout entire sequence history
        # TODO: can be optimized to be O(1) using `random.randint(1, num_items)` and checking if seen
        item_neg_pool = set(range(1, num_items + 1)) - set(user_seq)
        random_neg_seq = random.sample(list(item_neg_pool), k=len(target_seq))
        # move list of ints into numpy arrays
        input_seq_batch[i, :] = input_seq
        target_pos_seq_batch[i, :] = target_seq
        target_neg_seq_batch[i, :] = random_neg_seq
    return input_seq_batch, target_pos_seq_batch, target_neg_seq_batch


def main():
    parser = argparse.ArgumentParser(description="nanoSASRec train v1")
    parser.add_argument("--dataset", type=str, default="ml-1m",
                        choices=DATASETS,
                        help="Dataset to use")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Root data directory")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Number of examples per batch")
    parser.add_argument("--maxlen", type=int, default=50,
                        help="Max sequence length per user")
    args = parser.parse_args()

    # Load pre-processed dataset with train, validation, test splits
    user_train, user_valid, user_test, num_users, num_items, item_id_map = load_data(args.data_dir, args.dataset)
    input_seq, pos_seq, neg_seq = sample_batch(user_train, num_users=num_users, num_items=num_items, maxlen=args.maxlen, batch_size=args.batch_size)
    print(input_seq.shape, pos_seq.shape, neg_seq.shape)

if __name__ == "__main__":
    main()