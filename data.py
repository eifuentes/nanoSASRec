"""
nanoSASRec data preprocessing.

Downloads and preprocesses datasets for sequential recommendation.
Uses the exact same preprocessing as the BERT4Rec repository
(Petrov & Macdonald, RecSys 2022 - arxiv:2207.07483) to ensure
reproducibility with published numbers.

Protocol:
  - Treat ratings/reviews as implicit feedback (binary interaction)
  - Sort interactions by timestamp per user
  - k-core filtering: discard users and items with < 5 interactions
  - Leave-one-out split: last item = test, second-to-last = valid, rest = train

Usage:
  python data.py                    # downloads and preprocesses ML-1M
  python data.py --dataset ml-20m   # preprocesses MovieLens-20M
"""

import argparse
import json
import os
import zipfile
from collections import Counter, defaultdict
from itertools import chain
from urllib.request import urlretrieve
from typing import NamedTuple, Dict, List

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

URLS = {
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "ml-20m": "https://files.grouplens.org/datasets/movielens/ml-20m.zip",
}


def download(url: str, dest: str):
    """Download a file if it doesn't already exist."""
    if os.path.exists(dest):
        print(f"  already exists: {dest}")
        return
    print(f"  downloading {url} ...")
    urlretrieve(url, dest)
    print(f"  saved to {dest}")


def extract_zip(zip_path: str, extract_to: str):
    """Extract a zip file."""
    print(f"  extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)


# ---------------------------------------------------------------------------
# Raw data loaders - one per dataset
# Each returns list of (user_id, item_id, timestamp) tuples, raw strings ok
# ---------------------------------------------------------------------------


def load_ml1m_raw(data_dir: str):
    """Load MovieLens-1M raw interactions."""
    zip_path = os.path.join(data_dir, "ml-1m.zip")
    download(URLS["ml-1m"], zip_path)
    raw_dir = os.path.join(data_dir, "ml-1m")
    if not os.path.isdir(raw_dir):
        extract_zip(zip_path, data_dir)

    interactions = []
    ratings_file = os.path.join(raw_dir, "ratings.dat")
    with open(ratings_file, "r") as f:
        for line in f:
            user, item, _, ts = line.strip().split("::")
            interactions.append((user, item, int(ts)))
    return interactions


def load_ml20m_raw(data_dir: str):
    """Load MovieLens-20M raw interactions."""
    zip_path = os.path.join(data_dir, "ml-20m.zip")
    download(URLS["ml-20m"], zip_path)
    raw_dir = os.path.join(data_dir, "ml-20m")
    if not os.path.isdir(raw_dir):
        extract_zip(zip_path, data_dir)

    interactions = []
    ratings_file = os.path.join(raw_dir, "ratings.csv")
    with open(ratings_file, "r") as f:
        next(f)  # skip header
        for line in f:
            user, item, _, ts = line.strip().split(",")
            interactions.append((user, item, int(float(ts))))
    return interactions


def load_beauty_raw(data_dir: str):
    """
    Load Amazon Beauty interactions.

    NOTE: We use the preprocessed version from the BERT4Rec / SASRec repos.
    Download it manually from:
      https://github.com/FeiSun/BERT4Rec/tree/master/data
    Place the file as data/beauty/Beauty.txt
    Each line: user_id item_id1 item_id2 ...  (already sorted by time)
    """
    txt_path = os.path.join(data_dir, "beauty", "Beauty.txt")
    if not os.path.exists(txt_path):
        print("\n  Amazon Beauty requires manual download.")
        print("  Download from: https://github.com/FeiSun/BERT4Rec/tree/master/data")
        print(f"  Place the file at: {txt_path}")
        print("  Format: one line per user, space-separated item IDs in temporal order.\n")
        raise FileNotFoundError(f"Missing {txt_path}")

    # This file is already preprocessed: each line is user's item sequence
    # We return it in a special format that skips k-core filtering
    interactions = []
    with open(txt_path, "r") as f:
        for user_id, line in enumerate(f, 1):
            items = line.strip().split()
            for t, item_id in enumerate(items):
                interactions.append((str(user_id), item_id, t))
    return interactions


# ---------------------------------------------------------------------------
# k-core filtering
# (Kang & McAuley, ICDM 2018 - arxiv:1808.09781, Section IV-A)
# This is the methodological constant that makes numbers comparable
# across the entire SASRec paper lineage.
# ---------------------------------------------------------------------------


def kcore_filter(interactions: list, k: int = 5):
    """
    Iteratively remove users and items with fewer than k interactions
    until convergence.
    """
    while True:
        user_counts = Counter(u for u, i, t in interactions)
        item_counts = Counter(i for u, i, t in interactions)

        filtered = [(u, i, t) for u, i, t in interactions if user_counts[u] >= k and item_counts[i] >= k]

        if len(filtered) == len(interactions):
            break  # converged
        interactions = filtered

    return interactions


# ---------------------------------------------------------------------------
# Reindex and split
# ---------------------------------------------------------------------------


class DatasetSplit(NamedTuple):
    user_train: Dict[int, List[int]]
    user_train_ts: Dict[int, List[int]]
    user_valid: Dict[int, List[int]]
    user_valid_ts: Dict[int, List[int]]
    user_test: Dict[int, List[int]]
    user_test_ts: Dict[int, List[int]]
    num_users: int
    num_items: int
    item_map: Dict


def reindex_and_split(interactions: list) -> DatasetSplit:
    """
    Reindex users/items to contiguous integers starting from 1.
    Item 0 is reserved as the padding token.

    Then apply leave-one-out temporal split:
      - last item per user -> test
      - second-to-last   -> valid
      - rest              -> train

    Returns:
      user_train: dict[int, list[int]]  - user -> list of item ids
      user_valid: dict[int, list[int]]  - user -> [item_id] or []
      user_test:  dict[int, list[int]]  - user -> [item_id] or []
      num_users:  int
      num_items:  int
    """
    # sort by (user, timestamp) to get temporal order
    # NOTE: ensures each users history is sorted from oldest to newest
    interactions.sort(key=lambda x: (x[0], x[2]))

    # build sequences per user (preserving temporal order)
    user_seqs_raw = defaultdict(list)
    user_ts_raw = defaultdict(list)
    for u, i, t in interactions:
        user_seqs_raw[u].append(i)
        user_ts_raw[u].append(t)

    # reindex users and items to contiguous integers starting from 1
    # (0 is reserved as the padding token for items)
    user_map = {u: idx for idx, u in enumerate(sorted(user_seqs_raw), 1)}
    all_items = set(chain.from_iterable(user_seqs_raw.values()))
    item_map = {i: idx for idx, i in enumerate(sorted(all_items), 1)}

    # count number of items and users
    num_users = len(user_seqs_raw)
    num_items = len(item_map)

    # leave-one-out split
    # (Kang & McAuley 2018, Section IV-A: "the most recent action for testing,
    #  the second most recent action for validation, and all remaining for training")
    user_train = {}
    user_train_ts = {}
    user_valid = {}
    user_valid_ts = {}
    user_test = {}
    user_test_ts = {}

    # apply reindexing and split
    for user_raw, seq in user_seqs_raw.items():
        # user sequences mapped and associated timestamps
        user = user_map[user_raw]
        # NOTE: timestamps are unix epoch integers (seconds since 1970) — they don't need
        # reindexing like user/item IDs. They carry forward as-is through the split.
        ts = user_ts_raw[user_raw]
        # conduct a split
        if len(seq) < 3:
            # not enough interactions for a proper split
            # split for timestamps
            user_train_ts[user] = ts
            user_valid_ts[user] = []
            user_test_ts[user] = []
            # split for sequences
            user_train[user] = [item_map[i] for i in seq]
            user_valid[user] = []
            user_test[user] = []
        else:
            # split for timestamps
            user_train_ts[user] = ts[:-2]
            user_valid_ts[user] = [ts[-2]]
            user_test_ts[user] = [ts[-1]]
            # split for sequences
            user_train[user] = [item_map[i] for i in seq[:-2]]
            user_valid[user] = [item_map[seq[-2]]]
            user_test[user] = [item_map[seq[-1]]]

    # item_map: raw_item_id -> reindexed_id
    # callers can invert this to map reindexed IDs back to original IDs
    return DatasetSplit(
        user_train=user_train,
        user_train_ts=user_train_ts,
        user_valid=user_valid,
        user_valid_ts=user_valid_ts,
        user_test=user_test,
        user_test_ts=user_test_ts,
        num_users=num_users,
        num_items=num_items,
        item_map=item_map,
    )


# ---------------------------------------------------------------------------
# Write to canonical format
# ---------------------------------------------------------------------------


def write_txt(dataset: DatasetSplit, out_path: str):
    """
    Write to the canonical SASRec .txt format:
      user_id item_id_1 item_id_2 ... item_id_n

    This is the FULL sequence per user (train + valid + test concatenated).
    The split is implicit: last = test, second-to-last = valid, rest = train.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for u in range(1, dataset.num_users + 1):
            full_seq = dataset.user_train.get(u, []) + dataset.user_valid.get(u, []) + dataset.user_test.get(u, [])
            items_str = " ".join(str(i) for i in full_seq)
            f.write(f"{u} {items_str}\n")

    print(f"\n  Wrote {out_path}")
    print(f"  Users: {dataset.num_users}, Items: {dataset.num_items}")

    # compute stats
    total_actions = sum(
        len(dataset.user_train.get(u, [])) + len(dataset.user_valid.get(u, [])) + len(dataset.user_test.get(u, []))
        for u in range(1, dataset.num_users + 1)
    )
    print(f"  Total interactions: {total_actions}")
    print(f"  Avg sequence length: {total_actions / dataset.num_users:.1f}")


# ---------------------------------------------------------------------------
# Write the split as JSON for train_v*.py to load
# ---------------------------------------------------------------------------


def write_split(dataset: DatasetSplit, out_dir: str):
    """Write the split and item ID mapping as JSON for train_v*.py to load."""
    os.makedirs(out_dir, exist_ok=True)

    # reverse map: reindexed_id (int) -> raw_item_id (str)
    # this lets train scripts map predictions back to original item IDs
    reverse_item_map = {str(v): k for k, v in dataset.item_map.items()}

    split = {
        "user_train": {str(k): v for k, v in dataset.user_train.items()},
        "user_valid": {str(k): v for k, v in dataset.user_valid.items()},
        "user_test": {str(k): v for k, v in dataset.user_test.items()},
        "user_train_timestamps": {str(k): v for k, v in dataset.user_train_ts.items()},
        "user_valid_timestamps": {str(k): v for k, v in dataset.user_valid_ts.items()},
        "user_test_timestamps": {str(k): v for k, v in dataset.user_test_ts.items()},
        "num_users": dataset.num_users,
        "num_items": dataset.num_items,
        "item_id_map": reverse_item_map,
    }

    split_path = os.path.join(out_dir, "split.json")
    with open(split_path, "w") as f:
        json.dump(split, f)
    print(f"  Wrote split to {split_path}")


def load_split(data_dir: str):
    """Load a preprocessed split from disk. Used by train_v*.py scripts."""
    split_path = os.path.join(data_dir, "split.json")
    with open(split_path, "r") as f:
        split = json.load(f)

    user_train = {int(k): v for k, v in split["user_train"].items()}
    user_valid = {int(k): v for k, v in split["user_valid"].items()}
    user_test = {int(k): v for k, v in split["user_test"].items()}
    num_users = split["num_users"]
    num_items = split["num_items"]

    # item_id_map: reindexed_id (int) -> raw_item_id (str)
    item_id_map = {int(k): v for k, v in split.get("item_id_map", {}).items()}

    return user_train, user_valid, user_test, num_users, num_items, item_id_map


# ---------------------------------------------------------------------------
# Item titles (optional, for human-readable predictions)
# ---------------------------------------------------------------------------


def write_item_titles_ml1m(data_dir: str, out_dir: str):
    """Parse ML-1M movies.dat and write a titles.json mapping raw movie ID -> title."""
    movies_file = os.path.join(data_dir, "ml-1m", "movies.dat")
    if not os.path.exists(movies_file):
        return

    titles = {}
    with open(movies_file, "r", encoding="latin-1") as f:
        for line in f:
            # format: MovieID::Title::Genres
            parts = line.strip().split("::")
            movie_id, title = parts[0], parts[1]
            titles[movie_id] = title

    titles_path = os.path.join(out_dir, "titles.json")
    with open(titles_path, "w") as f:
        json.dump(titles, f)
    print(f"  Wrote {len(titles)} item titles to {titles_path}")


def write_item_titles_ml20m(data_dir: str, out_dir: str):
    """Parse ML-20M movies.csv and write a titles.json mapping raw movie ID -> title."""
    movies_file = os.path.join(data_dir, "ml-20m", "movies.csv")
    if not os.path.exists(movies_file):
        return

    titles = {}
    with open(movies_file, "r", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            # format: movieId,title,genres (title may contain commas, but is not quoted consistently)
            # safe approach: split on first and last comma
            parts = line.strip().split(",")
            movie_id = parts[0]
            # title is everything between first and last comma
            title = ",".join(parts[1:-1])
            titles[movie_id] = title

    titles_path = os.path.join(out_dir, "titles.json")
    with open(titles_path, "w") as f:
        json.dump(titles, f)
    print(f"  Wrote {len(titles)} item titles to {titles_path}")


def load_item_titles(data_dir: str):
    """Load item titles from disk. Returns dict: reindexed_id -> title string, or empty dict."""
    # first load the ID mapping (reindexed -> raw)
    split_path = os.path.join(data_dir, "split.json")
    titles_path = os.path.join(data_dir, "titles.json")

    if not os.path.exists(titles_path):
        return {}

    with open(split_path, "r") as f:
        split = json.load(f)
    with open(titles_path, "r") as f:
        raw_titles = json.load(f)

    item_id_map = split.get("item_id_map", {})

    # compose: reindexed_id -> raw_id -> title
    titles = {}
    for reindexed_id_str, raw_id in item_id_map.items():
        if raw_id in raw_titles:
            titles[int(reindexed_id_str)] = raw_titles[raw_id]

    return titles


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DATASETS = {
    "ml-1m": {
        "loader": load_ml1m_raw,
        "kcore": True,
        "titles_writer": write_item_titles_ml1m,
        # Expected stats after preprocessing (SASRec paper, Table II):
        #   Users: 6,040  Items: 3,416  Avg actions/user: 163.5
    },
    "beauty": {
        "loader": load_beauty_raw,
        "kcore": False,  # already preprocessed
        "titles_writer": None,
        # Expected stats (CL4SRec, 2010.14395):
        #   Users: 22,363  Items: 12,101  Avg actions/user: 8.8
    },
    "ml-20m": {
        "loader": load_ml20m_raw,
        "kcore": True,
        "titles_writer": write_item_titles_ml20m,
        # Expected stats (Meta GR repo):
        #   Users: ~138K  Items: ~27K  Avg actions/user: ~144
    },
}


def main():
    parser = argparse.ArgumentParser(description="nanoSASRec data preprocessing")
    parser.add_argument(
        "--dataset", type=str, default="ml-1m", choices=list(DATASETS.keys()), help="Dataset to preprocess"
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory")
    args = parser.parse_args()

    ds = DATASETS[args.dataset]
    print(f"\n{'=' * 60}")
    print(f"  Preprocessing: {args.dataset}")
    print(f"{'=' * 60}")

    # step 1: load raw interactions
    print("\n[1/4] Loading raw data ...")
    os.makedirs(args.data_dir, exist_ok=True)
    raw = ds["loader"](args.data_dir)
    print(f"  Raw interactions: {len(raw)}")

    # step 2: k-core filtering (if needed)
    if ds["kcore"]:
        print("\n[2/4] Applying k=5 core filtering ...")
        filtered = kcore_filter(raw, k=5)
        print(f"  After filtering: {len(filtered)} (removed {len(raw) - len(filtered)})")
    else:
        print("\n[2/4] Skipping k-core (pre-filtered dataset)")
        filtered = raw

    # step 3: reindex and split
    print("\n[3/4] Reindexing and splitting ...")
    dataset = reindex_and_split(filtered)

    # step 4: write outputs
    out_dir = os.path.join(args.data_dir, args.dataset)
    txt_path = os.path.join(out_dir, f"{args.dataset}.txt")
    print("\n[4/4] Writing outputs ...")
    write_txt(dataset, txt_path)
    write_split(dataset, out_dir)

    # write item titles if available (for human-readable predictions)
    titles_writer = ds.get("titles_writer")
    if titles_writer:
        titles_writer(args.data_dir, out_dir)

    print(f"\n{'=' * 60}")
    print("  Done! Run train v1 through v8 to start training.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
