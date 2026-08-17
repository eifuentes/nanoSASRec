"""
nanoSASRec data preprocessing.

Downloads and preprocesses datasets for sequential recommendation.
MovieLens and Beauty follow the SASRec/BERT4Rec preprocessing lineage
(Petrov & Macdonald, RecSys 2022 - arxiv:2207.07483). KuaiRand uses
documented recipe flags plus a DuckDB/Parquet cache because its raw logs are
large enough that repeated CSV scans make exploration painfully slow.

MovieLens/Beauty protocol:
  - Treat ratings/reviews as implicit feedback (binary interaction)
  - Sort interactions by timestamp per user
  - k-core filtering: discard users and items with < 5 interactions
  - Leave-one-out split: last item = test, second-to-last = valid, rest = train

KuaiRand protocol:
  - Convert standard logs to typed Parquet once
  - Materialize an interaction rule, e.g. all impressions for Pure or
    long_view/not_hate/tab=1 for the 27K long-sequence slice
  - Preserve repeated impressions
  - Apply configured item/user frequency floors to convergence
  - Use the same leave-one-out split format as the rest of the repo

Usage:
  python data.py                    # downloads and preprocesses ML-1M
  python data.py --dataset ml-20m   # preprocesses MovieLens-20M
  python data.py --dataset kuairand-pure  # confirmed Pure/Gamma all-impression slice
  python data.py --dataset kuairand-27k --kuairand-interaction-rule long-view-not-hate-tab1 --kuairand-exclude-rand --kuairand-item-floor 555 --kuairand-user-floor 5
  python data.py --dataset kuairand-27k --kuairand-analysis-only  # stats/profile only
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import tarfile
import time
import zipfile
from collections import Counter, defaultdict
from itertools import chain
from typing import Callable, Dict, List, NamedTuple
from urllib.request import Request, urlopen, urlretrieve

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

URLS = {
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "ml-20m": "https://files.grouplens.org/datasets/movielens/ml-20m.zip",
}

KUAIRAND_ZENODO_BASE = "https://zenodo.org/records/10439422/files"
KUAIRAND_VARIANTS = {
    "pure": ("KuaiRand-Pure.tar.gz", "0820331067a3784d9691136f772b35a7"),
    "27k": ("KuaiRand-27K.tar.gz", "3e3c799a24e2d23a4d2c757fbf9adf59"),
}
KUAIRAND_LOG_FILES = {
    "pure": (
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
    ),
    "27k": (
        "log_standard_4_08_to_4_21_27k_part1.csv",
        "log_standard_4_08_to_4_21_27k_part2.csv",
        "log_standard_4_22_to_5_08_27k_part1.csv",
        "log_standard_4_22_to_5_08_27k_part2.csv",
    ),
}

# KuaiRand defaults reproduce the confirmed Pure/FuXi-gamma slice. The 27K
# long-sequence recipe is selected explicitly with CLI flags; see README.md.
KUAIRAND_ITEM_FLOOR = 1
KUAIRAND_USER_FLOOR = 5
KUAIRAND_INTERACTION_RULE_VERSION = "all-impressions-v1"
KUAIRAND_DEFAULT_INTERACTION_RULE = "all-impressions"

# KuaiRand interaction rules are named combinations of simple row predicates.
# Keeping the rule vocabulary here lets the Python smoke-test helper and the
# DuckDB materialization SQL share the same intent.
KUAIRAND_INTERACTION_RULE_CONDITIONS = {
    "all-impressions": (),
    "long-view-not-hate": ("long_view", "not_hate"),
    "long-view-not-hate-tab1": ("long_view", "not_hate", "tab1"),
    "click": ("click",),
    "click-not-hate": ("click", "not_hate"),
    "click-tab1": ("click", "tab1"),
    "click-not-hate-tab1": ("click", "not_hate", "tab1"),
}
KUAIRAND_INTERACTION_RULES = tuple(KUAIRAND_INTERACTION_RULE_CONDITIONS)
KUAIRAND_INTERACTION_RULE_VERSIONS = {rule: f"{rule}-v1" for rule in KUAIRAND_INTERACTION_RULES}
KUAIRAND_INTERACTION_RULE_VERSIONS["all-impressions"] = KUAIRAND_INTERACTION_RULE_VERSION
KUAIRAND_CONDITION_SQL = {
    "click": "is_click = 1",
    "long_view": "long_view = 1",
    "not_hate": "is_hate != 1",
    "not_rand": "is_rand != 1",
    "tab1": "tab = 1",
}
KUAIRAND_PARQUET_CACHE_VERSION = "standard-parquet-v3-content-hash-canonical-order"
KUAIRAND_FILTER_IMPLEMENTATION_VERSION = "duckdb-parquet-filter-v1"
KUAIRAND_ITEM_COUNT_CACHE_VERSION = "duckdb-staged-item-counts-v1"

# Diagnostic target from FuXi-alpha Table 1. We keep this only to print deltas
# for reproducibility notes; these values do not define the maintained 27K
# recipe, which is a principled long-view engagement slice.
KUAIRAND_FUXI_ALPHA_STATS = {
    "users": 25_634,
    "items": 7_550,
    "interactions": 6_945_823,
    "avg_length": 270.96,
}
KUAIRAND_DOWNLOAD_CHECKPOINT_BYTES = 1 << 30
# Remote preprocessing interface: remote.py may set this callback to checkpoint
# a Modal Volume after each expensive KuaiRand preprocessing artifact is durable.
# Keep this module-level hook stable unless remote.py is updated with it.
KUAIRAND_STAGE_FILE_CALLBACK: Callable[[str], None] | None = None
KUAIRAND_DOWNLOAD_CHECKPOINT_CALLBACK: Callable[[str], None] | None = None
KUAIRAND_ITEM_COUNT_CACHE_CALLBACK: Callable[[str], None] | None = None
KUAIRAND_LAST_RUN_METADATA: dict | None = None


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


def _print_download_progress(downloaded: int, total: int):
    if total <= 0:
        print(f"\r  downloading ... {downloaded / 1e9:.2f} GB", end="", flush=True)
        return
    print(
        f"\r  downloading ... {100 * downloaded / total:5.1f}% ({downloaded / 1e9:.2f}/{total / 1e9:.2f} GB)",
        end="",
        flush=True,
    )


def verify_md5(path: str, expected: str):
    """Verify a downloaded archive against KuaiRand's published checksum."""
    digest = hashlib.md5()  # noqa: S324 - MD5 is the publisher-provided integrity checksum
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"MD5 mismatch for {path}: got {actual}, expected {expected}; remove and re-download it")


def _sha256_file(path: str) -> str:
    """Return a content digest for a source file used by a derived-data cache."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_range_total(content_range: str | None) -> int | None:
    """Parse a Content-Range header like 'bytes 10-99/100'."""
    if not content_range or "/" not in content_range:
        return None
    total = content_range.rsplit("/", 1)[1]
    if total == "*":
        return None
    try:
        return int(total)
    except ValueError:
        return None


def _checkpoint_kuairand_download(label: str):
    if KUAIRAND_DOWNLOAD_CHECKPOINT_CALLBACK is not None:
        KUAIRAND_DOWNLOAD_CHECKPOINT_CALLBACK(label)


def _download_kuairand_archive_once(url: str, partial: str) -> None:
    existing_bytes = os.path.getsize(partial) if os.path.exists(partial) else 0
    request = Request(url)
    if existing_bytes:
        request.add_header("Range", f"bytes={existing_bytes}-")
        print(f"  resuming partial archive at {existing_bytes / 1e9:.2f} GB ...")

    with urlopen(request) as response, open(partial, "ab" if existing_bytes else "wb") as target:
        status = response.getcode()
        if existing_bytes and status != 206:
            print("  server did not honor resume request; restarting archive download ...")
            existing_bytes = 0
            target.seek(0)
            target.truncate()

        content_length = int(response.headers.get("Content-Length") or 0)
        total = _content_range_total(response.headers.get("Content-Range")) or existing_bytes + content_length
        downloaded = existing_bytes
        next_checkpoint = ((downloaded // KUAIRAND_DOWNLOAD_CHECKPOINT_BYTES) + 1) * KUAIRAND_DOWNLOAD_CHECKPOINT_BYTES

        for chunk in iter(lambda: response.read(8 << 20), b""):
            target.write(chunk)
            downloaded += len(chunk)
            _print_download_progress(downloaded, total)
            if downloaded >= next_checkpoint:
                target.flush()
                os.fsync(target.fileno())
                print()
                _checkpoint_kuairand_download(f"{os.path.basename(partial)} at {downloaded / 1e9:.2f} GB")
                next_checkpoint += KUAIRAND_DOWNLOAD_CHECKPOINT_BYTES
    print()


def _download_kuairand_archive(url: str, dest: str, expected_md5: str):
    """Download atomically with resume support, then verify before use."""
    if os.path.exists(dest):
        print(f"  verifying existing archive: {dest}")
        verify_md5(dest, expected_md5)
        return

    partial = f"{dest}.partial"
    print(f"  downloading {url} ...")
    try:
        _download_kuairand_archive_once(url, partial)
        verify_md5(partial, expected_md5)
        os.replace(partial, dest)
        _checkpoint_kuairand_download(os.path.basename(dest))
    except Exception:
        # Keep the partial download for diagnosis/resume, but it will never be
        # mistaken for a verified archive on the next run.
        print(f"\n  download incomplete; resumable partial file left at {partial}")
        raise


def extract_selected_tar(tar_path: str, output_dir: str, required_files: tuple[str, ...]):
    """Safely stream only named log files from a tar archive.

    Members are flattened to their known basenames. No archive-controlled path
    is ever used as a destination, preventing path traversal and avoiding
    extraction of KuaiRand's large, unused feature files.
    """
    os.makedirs(output_dir, exist_ok=True)
    required = set(required_files)
    present = {name for name in required if os.path.isfile(os.path.join(output_dir, name))}
    if present == required:
        return

    found = set(present)
    print(f"  extracting {len(required - present)} required log file(s) ...")
    with tarfile.open(tar_path, "r|gz") as archive:
        for member in archive:
            basename = os.path.basename(member.name)
            if basename not in required or basename in found or not member.isfile():
                continue
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Could not read {member.name} from {tar_path}")
            destination = os.path.join(output_dir, basename)
            partial = f"{destination}.partial"
            with source, open(partial, "wb") as target:
                shutil.copyfileobj(source, target, length=1 << 20)
            os.replace(partial, destination)
            found.add(basename)
            print(f"  extracted {basename}")
            if found == required:
                break

    missing = sorted(required - found)
    if missing:
        raise FileNotFoundError(f"Archive {tar_path} is missing required log files: {missing}")


def download_kuairand(variant: str, data_dir: str, delete_archive_after_extract: bool = False) -> list[str]:
    """Download and selectively extract a KuaiRand variant's standard logs."""
    archive_name, checksum = KUAIRAND_VARIANTS[variant]
    archive_path = os.path.join(data_dir, archive_name)
    raw_dir = os.path.join(data_dir, archive_name.removesuffix(".tar.gz"), "data")
    required = KUAIRAND_LOG_FILES[variant]

    paths = [os.path.join(raw_dir, name) for name in required]
    if not all(os.path.isfile(path) for path in paths):
        os.makedirs(data_dir, exist_ok=True)
        _download_kuairand_archive(f"{KUAIRAND_ZENODO_BASE}/{archive_name}", archive_path, checksum)
        extract_selected_tar(archive_path, raw_dir, required)
        if delete_archive_after_extract and os.path.exists(archive_path):
            os.remove(archive_path)
            print(f"  deleted archive after extraction: {archive_path}")

    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"Missing required KuaiRand logs: {missing}")
    if delete_archive_after_extract and os.path.exists(archive_path):
        os.remove(archive_path)
        print(f"  deleted archive after extraction: {archive_path}")
    return paths


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


def _kuairand_number_not_one(value: str | None) -> bool:
    """Fast path integer-like KuaiRand flags, falling back for rare decimals."""
    if value is None:
        return False
    try:
        return int(value) != 1
    except (TypeError, ValueError):
        try:
            return int(float(value)) != 1
        except (TypeError, ValueError):
            return False


def _kuairand_number_is_one(value: str | None) -> bool:
    """Fast path integer-like KuaiRand flags, falling back for rare decimals."""
    try:
        return int(value) == 1
    except (TypeError, ValueError):
        try:
            return int(float(value)) == 1
        except (TypeError, ValueError):
            return False


def _kuairand_import_duckdb():
    try:
        import duckdb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "KuaiRand preprocessing now requires DuckDB. Install project dependencies with `uv sync`, "
            "or run through Modal after updating uv.lock."
        ) from exc
    return duckdb


def _kuairand_interaction_rule_version(interaction_rule: str) -> str:
    return KUAIRAND_INTERACTION_RULE_VERSIONS[interaction_rule]


def _kuairand_recipe_conditions(
    interaction_rule: str,
    exclude_hate: bool = False,
    exclude_rand: bool = False,
) -> list[str]:
    """Return the row-level predicates that define a KuaiRand recipe."""
    conditions = list(KUAIRAND_INTERACTION_RULE_CONDITIONS[interaction_rule])
    for condition, enabled in (("not_hate", exclude_hate), ("not_rand", exclude_rand)):
        if enabled and condition not in conditions:
            conditions.append(condition)
    return conditions


def _kuairand_condition_matches(
    condition: str,
    is_hate: str,
    long_view: str | None,
    is_click: str | None,
    tab: str | None,
    is_rand: str | None,
) -> bool:
    if condition == "click":
        return _kuairand_number_is_one(is_click)
    if condition == "long_view":
        return _kuairand_number_is_one(long_view)
    if condition == "not_hate":
        return _kuairand_number_not_one(is_hate)
    if condition == "not_rand":
        return _kuairand_number_not_one(is_rand)
    if condition == "tab1":
        return _kuairand_number_is_one(tab)
    raise ValueError(f"Unsupported KuaiRand recipe condition: {condition}")


def kuairand_include_interaction(
    is_hate: str,
    exclude_hate: bool = False,
    interaction_rule: str = KUAIRAND_DEFAULT_INTERACTION_RULE,
    long_view: str | None = None,
    is_click: str | None = None,
    tab: str | None = None,
    is_rand: str | None = None,
    exclude_rand: bool = False,
) -> bool:
    """Return whether a KuaiRand row belongs to the selected interaction recipe.

    Production preprocessing materializes rules in DuckDB. This small Python
    mirror lets tests exercise the same rule vocabulary without building a
    database. Keep it in sync with KUAIRAND_CONDITION_SQL.
    """
    return all(
        _kuairand_condition_matches(
            condition,
            is_hate=is_hate,
            long_view=long_view,
            is_click=is_click,
            tab=tab,
            is_rand=is_rand,
        )
        for condition in _kuairand_recipe_conditions(interaction_rule, exclude_hate, exclude_rand)
    )


def _duckdb_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _duckdb_string_list(values: list[str]) -> str:
    return "[" + ", ".join(_duckdb_quote(value) for value in values) + "]"


def _kuairand_processed_dir(data_dir: str, variant: str) -> str:
    return os.path.join(data_dir, f"kuairand-{variant}")


def _kuairand_duckdb_path(processed_dir: str) -> str:
    return os.path.join(processed_dir, "kuairand.duckdb")


def _kuairand_parquet_dir(processed_dir: str) -> str:
    return os.path.join(processed_dir, "standard-parquet")


def _kuairand_parquet_path(processed_dir: str, source_name: str) -> str:
    stem = source_name.removesuffix(".csv")
    return os.path.join(_kuairand_parquet_dir(processed_dir), f"{stem}.parquet")


def _kuairand_profile_path(
    processed_dir: str,
    exclude_hate: bool,
    interaction_rule: str = KUAIRAND_DEFAULT_INTERACTION_RULE,
    exclude_rand: bool = False,
) -> str:
    suffixes = []
    if interaction_rule != KUAIRAND_DEFAULT_INTERACTION_RULE:
        suffixes.append(interaction_rule)
    if exclude_hate:
        suffixes.append("exclude-hate")
    if exclude_rand:
        suffixes.append("exclude-rand")
    if not suffixes:
        return os.path.join(processed_dir, "item_frequency_profile.csv")
    return os.path.join(processed_dir, f"item_frequency_profile-{'-'.join(suffixes)}.csv")


def _kuairand_raw_parquet_fingerprint(log_paths: list[str], variant: str) -> str:
    """Identify the raw standard-log inputs that back the Parquet cache."""
    archive_name, archive_md5 = KUAIRAND_VARIANTS[variant]
    source = []
    for path in log_paths:
        stat = os.stat(path)
        source.append((os.path.basename(path), stat.st_size, _sha256_file(path)))
    payload = json.dumps(
        {
            "variant": variant,
            "archive": archive_name,
            "archive_md5": archive_md5,
            "parquet_cache_version": KUAIRAND_PARQUET_CACHE_VERSION,
            "source": source,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _kuairand_stage_fingerprint(
    raw_parquet_fingerprint: str,
    interaction_rule: str,
    exclude_hate: bool,
    exclude_rand: bool,
) -> str:
    payload = json.dumps(
        {
            "raw_parquet_fingerprint": raw_parquet_fingerprint,
            "interaction_rule": _kuairand_interaction_rule_version(interaction_rule),
            "exclude_hate": exclude_hate,
            "exclude_rand": exclude_rand,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _kuairand_filter_fingerprint(stage_fingerprint: str, item_floor: int, user_floor: int) -> str:
    """Identify the cheap, sweepable recipe derived from rule-aware staging."""
    payload = json.dumps(
        {
            "stage_fingerprint": stage_fingerprint,
            "filter_implementation": KUAIRAND_FILTER_IMPLEMENTATION_VERSION,
            "item_floor": item_floor,
            "user_floor": user_floor,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def kuairand_recipe_slug(
    item_floor: int,
    user_floor: int,
    exclude_hate: bool = False,
    interaction_rule: str = KUAIRAND_DEFAULT_INTERACTION_RULE,
    exclude_rand: bool = False,
) -> str:
    """Stable path-friendly KuaiRand recipe label."""
    parts = []
    if interaction_rule != KUAIRAND_DEFAULT_INTERACTION_RULE:
        parts.extend(["rule", interaction_rule])
    parts.extend([f"item{item_floor}", f"user{user_floor}"])
    if exclude_hate:
        parts.append("exclude-hate")
    if exclude_rand:
        parts.append("exclude-rand")
    return "-".join(parts)


def kuairand_output_dataset_name(dataset: str, args: argparse.Namespace) -> str:
    """Return the output dataset directory name for a KuaiRand recipe."""
    if not dataset.startswith("kuairand-") or args.kuairand_flat_output:
        return dataset
    return (
        f"{dataset}-"
        f"{kuairand_recipe_slug(args.kuairand_item_floor, args.kuairand_user_floor, args.kuairand_exclude_hate, args.kuairand_interaction_rule, args.kuairand_exclude_rand)}"
    )


def output_txt_filename(output_dataset: str) -> str:
    """Name the canonical .txt after the actual output dataset directory."""
    return f"{output_dataset}.txt"


def write_kuairand_recipe_metadata(out_dir: str, dataset: str, args: argparse.Namespace):
    """Record the KuaiRand preprocessing recipe next to split.json."""
    if not dataset.startswith("kuairand-"):
        return
    recipe = {
        "dataset": dataset,
        "output_dataset": os.path.basename(out_dir),
        "item_floor": args.kuairand_item_floor,
        "user_floor": args.kuairand_user_floor,
        "interaction_rule": _kuairand_interaction_rule_version(args.kuairand_interaction_rule),
        "exclude_hate": args.kuairand_exclude_hate,
        "exclude_rand": args.kuairand_exclude_rand,
        "filter_implementation": KUAIRAND_FILTER_IMPLEMENTATION_VERSION,
        "tab_scope_note": "tab=1 is an empirical scope filter; KuaiRand does not publish tab value semantics.",
        "tie_break": "ORDER BY user_id, time_ms, item_id, source_file; equal millisecond tails use stable item identity and source file",
    }
    if KUAIRAND_LAST_RUN_METADATA is not None:
        recipe["run_metadata"] = KUAIRAND_LAST_RUN_METADATA
    path = os.path.join(out_dir, "recipe.json")
    with open(path, "w") as f:
        json.dump(recipe, f, indent=2)
    print(f"  Wrote KuaiRand recipe metadata to {path}")


def _kuairand_connect(database_path: str):
    duckdb = _kuairand_import_duckdb()
    os.makedirs(os.path.dirname(database_path), exist_ok=True)
    connection = duckdb.connect(database_path)
    connection.execute("SET preserve_insertion_order = true")
    connection.execute("CREATE TABLE IF NOT EXISTS metadata (key VARCHAR PRIMARY KEY, value VARCHAR NOT NULL)")
    return connection


def _kuairand_metadata(connection) -> dict[str, str]:
    return dict(connection.execute("SELECT key, value FROM metadata").fetchall())


def _kuairand_set_metadata(connection, pairs: dict[str, str | int | float | bool | dict | list | None]):
    rows = []
    for key, value in pairs.items():
        if isinstance(value, str):
            serialized = value
        else:
            serialized = json.dumps(value)
        rows.append((key, serialized))
    connection.executemany("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", rows)


def _kuairand_has_table(connection, table: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_schema = 'main' AND table_name = ?",
            (table,),
        ).fetchone()
        is not None
    )


def _kuairand_checkpoint_database(connection):
    """Flush DuckDB's durable state before asking Modal to commit a volume."""
    connection.execute("CHECKPOINT")


def _kuairand_parquet_file_counts(connection, parquet_path: str) -> dict[str, int]:
    rows, hate_interactions, rand_interactions = connection.execute(
        f"""
        SELECT
            COUNT(*),
            SUM(CASE WHEN is_hate = 1 THEN 1 ELSE 0 END),
            SUM(CASE WHEN is_rand = 1 THEN 1 ELSE 0 END)
        FROM read_parquet({_duckdb_quote(parquet_path)})
        """
    ).fetchone()
    return {
        "rows": int(rows or 0),
        "hate_interactions": int(hate_interactions or 0),
        "rand_interactions": int(rand_interactions or 0),
    }


def _kuairand_required_csv_columns() -> set[str]:
    return {
        "user_id",
        "video_id",
        "time_ms",
        "date",
        "is_hate",
        "long_view",
        "is_click",
        "is_like",
        "is_follow",
        "is_comment",
        "is_forward",
        "is_rand",
        "tab",
        "play_time_ms",
        "duration_ms",
    }


def _kuairand_csv_columns_sql(header: list[str]) -> str:
    """Return a full DuckDB CSV schema.

    DuckDB's `columns={...}` argument must describe the whole CSV, not merely
    the projected columns. We type the fields used by this pipeline and read
    unused extras as VARCHAR so future KuaiRand releases or unused feature
    columns do not break conversion.
    """
    required = _kuairand_required_csv_columns()
    fields = []
    for column in header:
        duckdb_type = "BIGINT" if column in required else "VARCHAR"
        fields.append(f"{_duckdb_quote(column)}: {_duckdb_quote(duckdb_type)}")
    return "{" + ", ".join(fields) + "}"


def _validate_kuairand_csv_columns(path: str) -> list[str]:
    required = _kuairand_required_csv_columns()
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)
    if header is None:
        raise ValueError(f"{path} is empty")
    missing = required - set(header)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return header


def _ensure_kuairand_raw_parquet(connection, log_paths: list[str], variant: str, processed_dir: str) -> str:
    """Convert standard KuaiRand CSV logs to durable typed Parquet once.

    DuckDB can scan CSV directly, but 27K has 322M standard-log rows and this
    branch repeatedly explores rules/floors. A one-time typed Parquet cache pays
    off quickly through column projection, filter pushdown, and no repeated CSV
    parsing. Each source CSV becomes one Parquet file so Modal checkpoints have
    natural resume boundaries.
    """
    fingerprint = _kuairand_raw_parquet_fingerprint(log_paths, variant)
    parquet_dir = _kuairand_parquet_dir(processed_dir)
    expected_paths = [_kuairand_parquet_path(processed_dir, os.path.basename(path)) for path in log_paths]
    metadata = _kuairand_metadata(connection)
    completed_files = set(json.loads(metadata.get("parquet_completed_files", "[]")))
    file_rows = json.loads(metadata.get("parquet_file_rows", "{}"))
    file_source_counts = json.loads(metadata.get("parquet_file_source_counts", "{}"))
    cache_ready = (
        metadata.get("raw_parquet_fingerprint") == fingerprint
        and metadata.get("raw_parquet_state") == "ready"
        and all(os.path.isfile(path) for path in expected_paths)
    )
    expected_names = [os.path.basename(path) for path in log_paths]
    if cache_ready and all(name in file_rows and name in file_source_counts for name in expected_names):
        print(f"  reusing raw standard-log Parquet cache: {parquet_dir}")
        return fingerprint
    if cache_ready:
        print(f"  repairing raw standard-log Parquet metadata: {parquet_dir}")

    if metadata.get("raw_parquet_fingerprint") != fingerprint:
        if os.path.isdir(parquet_dir):
            shutil.rmtree(parquet_dir)
        completed_files = set()
        file_rows = {}
        file_source_counts = {}
        connection.execute("DELETE FROM metadata")
    os.makedirs(parquet_dir, exist_ok=True)
    _kuairand_set_metadata(
        connection,
        {
            "raw_parquet_fingerprint": fingerprint,
            "raw_parquet_state": "building",
            "parquet_cache_version": KUAIRAND_PARQUET_CACHE_VERSION,
        },
    )

    for path in log_paths:
        source_name = os.path.basename(path)
        parquet_path = _kuairand_parquet_path(processed_dir, source_name)
        if source_name in completed_files and os.path.isfile(parquet_path):
            if source_name in file_rows and source_name in file_source_counts:
                rows = int(file_rows[source_name])
            else:
                counts = _kuairand_parquet_file_counts(connection, parquet_path)
                rows = counts["rows"]
                file_source_counts[source_name] = {
                    "hate_interactions": counts["hate_interactions"],
                    "rand_interactions": counts["rand_interactions"],
                }
            file_rows[source_name] = rows
            print(f"  already converted to Parquet: {source_name}")
            continue

        header = _validate_kuairand_csv_columns(path)
        partial = f"{parquet_path}.partial"
        if os.path.exists(partial):
            os.remove(partial)
        print(f"  converting {source_name} to typed Parquet ...")
        quoted_path = _duckdb_quote(os.path.abspath(path))
        quoted_partial = _duckdb_quote(os.path.abspath(partial))
        quoted_source = _duckdb_quote(source_name)
        # time_ms is the event timestamp. For exact timestamp ties, item_id is
        # the stable video identity published by KuaiRand and source_file makes
        # the result deterministic across the dataset's split log files. Avoid
        # play_time_ms and duration_ms: they are feedback signals, not ordering
        # fields. Keeping the CSV scan parallel preserves conversion throughput.
        connection.execute(
            f"""
            COPY (
                SELECT
                    {quoted_source} AS source_file,
                    user_id,
                    video_id AS item_id,
                    time_ms,
                    date,
                    is_hate,
                    long_view,
                    is_click,
                    is_like,
                    is_follow,
                    is_comment,
                    is_forward,
                    is_rand,
                    tab,
                    play_time_ms,
                    duration_ms
                FROM read_csv(
                    {quoted_path},
                    header=true,
                    columns={_kuairand_csv_columns_sql(header)},
                    parallel=true,
                    ignore_errors=false
                )
            ) TO {quoted_partial} (FORMAT parquet, COMPRESSION zstd, COMPRESSION_LEVEL 1)
            """
        )
        os.replace(partial, parquet_path)
        counts = _kuairand_parquet_file_counts(connection, parquet_path)
        file_rows[source_name] = counts["rows"]
        file_source_counts[source_name] = {
            "hate_interactions": counts["hate_interactions"],
            "rand_interactions": counts["rand_interactions"],
        }
        completed_files.add(source_name)
        _kuairand_set_metadata(
            connection,
            {
                "parquet_completed_files": sorted(completed_files),
                "parquet_file_rows": file_rows,
                "parquet_file_source_counts": file_source_counts,
            },
        )
        print(f"    wrote {counts['rows']:,} rows to {parquet_path}")
        if KUAIRAND_STAGE_FILE_CALLBACK is not None:
            _kuairand_checkpoint_database(connection)
            KUAIRAND_STAGE_FILE_CALLBACK(source_name)

    source_counts = {
        "hate_interactions": sum(int(file_source_counts[name]["hate_interactions"]) for name in expected_names),
        "rand_interactions": sum(int(file_source_counts[name]["rand_interactions"]) for name in expected_names),
    }
    _kuairand_set_metadata(
        connection,
        {
            "parquet_completed_files": sorted(completed_files),
            "parquet_file_rows": file_rows,
            "parquet_file_source_counts": file_source_counts,
            "raw_parquet_state": "ready",
            "source_counts_fingerprint": fingerprint,
            "source_counts": source_counts,
        },
    )
    return fingerprint


def _kuairand_parquet_list(processed_dir: str, log_paths: list[str]) -> str:
    paths = [_kuairand_parquet_path(processed_dir, os.path.basename(path)) for path in log_paths]
    return _duckdb_string_list(paths)


def _kuairand_stage_where(interaction_rule: str, exclude_hate: bool, exclude_rand: bool) -> str:
    conditions = _kuairand_recipe_conditions(interaction_rule, exclude_hate, exclude_rand)
    clauses = [KUAIRAND_CONDITION_SQL[condition] for condition in conditions]
    return " AND ".join(clauses) if clauses else "TRUE"


def _ensure_kuairand_stage(
    connection,
    processed_dir: str,
    log_paths: list[str],
    raw_parquet_fingerprint: str,
    interaction_rule: str,
    exclude_hate: bool,
    exclude_rand: bool,
) -> tuple[str, bool]:
    """Materialize the active interaction recipe from raw Parquet into DuckDB.

    The raw Parquet cache is durable and rule-agnostic. The staged/filter tables
    intentionally represent one active recipe at a time; changing the rule or
    optional filters rebuilds these cheap derived tables without touching the
    Parquet files.
    """
    stage_fingerprint = _kuairand_stage_fingerprint(
        raw_parquet_fingerprint,
        interaction_rule,
        exclude_hate,
        exclude_rand,
    )
    metadata = _kuairand_metadata(connection)
    if (
        metadata.get("stage_fingerprint") == stage_fingerprint
        and metadata.get("stage_state") == "ready"
        and _kuairand_has_table(connection, "staged")
    ):
        print(f"  reusing DuckDB staged table: {_kuairand_duckdb_path(processed_dir)}")
        return stage_fingerprint, True

    print(
        f"  materializing DuckDB staged table for "
        f"interaction_rule={_kuairand_interaction_rule_version(interaction_rule)}, "
        f"exclude_hate={exclude_hate}, exclude_rand={exclude_rand} ..."
    )
    # Derived recipe tables are single-slot caches. This keeps the database easy
    # to inspect and avoids side-by-side copies of large filtered slices.
    connection.execute("DROP TABLE IF EXISTS filtered")
    connection.execute("DROP TABLE IF EXISTS staged_item_counts")
    connection.execute("DROP TABLE IF EXISTS staged")
    _kuairand_set_metadata(connection, {"stage_fingerprint": stage_fingerprint, "stage_state": "building"})
    parquet_list = _kuairand_parquet_list(processed_dir, log_paths)
    where_sql = _kuairand_stage_where(interaction_rule, exclude_hate, exclude_rand)
    connection.execute(
        f"""
        CREATE TABLE staged AS
        SELECT
            source_file,
            user_id,
            item_id,
            time_ms,
            date,
            is_hate,
            long_view,
            is_click,
            is_like,
            is_follow,
            is_comment,
            is_forward,
            is_rand,
            tab,
            play_time_ms,
            duration_ms
        FROM read_parquet({parquet_list})
        WHERE {where_sql}
        """
    )
    _kuairand_set_metadata(
        connection,
        {
            "stage_state": "ready",
            "interaction_rule": _kuairand_interaction_rule_version(interaction_rule),
            "exclude_hate": exclude_hate,
            "exclude_rand": exclude_rand,
        },
    )
    if KUAIRAND_STAGE_FILE_CALLBACK is not None:
        _kuairand_checkpoint_database(connection)
        KUAIRAND_STAGE_FILE_CALLBACK("duckdb-stage")
    return stage_fingerprint, False


def _kuairand_stats(connection, table: str = "filtered") -> dict[str, float | int]:
    if table not in {"staged", "filtered"}:
        raise ValueError(f"Unsupported KuaiRand stats table: {table}")
    interactions, users, items = connection.execute(
        f"SELECT COUNT(*), COUNT(DISTINCT user_id), COUNT(DISTINCT item_id) FROM {table}"
    ).fetchone()
    return {
        "users": int(users),
        "items": int(items),
        "interactions": int(interactions),
        "avg_length": (interactions / users) if users else 0.0,
    }


def _print_kuairand_stats(label: str, stats: dict[str, float | int]):
    print(
        f"  {label}: users={stats['users']:,}, items={stats['items']:,}, "
        f"interactions={stats['interactions']:,}, avg_length={stats['avg_length']:.2f}"
    )


def _kuairand_final_timestamp_tie_stats(connection) -> dict[str, float | int]:
    """Count users whose held-out tail is tied on the final timestamp."""
    users, tied_users, tied_interactions, max_tie = connection.execute(
        """
        WITH max_times AS (
            SELECT user_id, MAX(time_ms) AS max_time_ms
            FROM filtered
            GROUP BY user_id
        ),
        final_counts AS (
            SELECT f.user_id, COUNT(*) AS final_count
            FROM filtered AS f
            JOIN max_times AS m
              ON f.user_id = m.user_id AND f.time_ms = m.max_time_ms
            GROUP BY f.user_id
        )
        SELECT
            COUNT(*),
            COALESCE(SUM(CASE WHEN final_count > 1 THEN 1 ELSE 0 END), 0),
            COALESCE(SUM(CASE WHEN final_count > 1 THEN final_count ELSE 0 END), 0),
            COALESCE(MAX(final_count), 0)
        FROM final_counts
        """
    ).fetchone()
    return {
        "users": int(users),
        "tied_users": int(tied_users),
        "tied_user_pct": (100 * tied_users / users) if users else 0.0,
        "tied_interactions": int(tied_interactions),
        "max_tie": int(max_tie),
    }


def _print_kuairand_tie_stats(stats: dict[str, float | int]):
    print(
        "  final timestamp ties: "
        f"users={stats['tied_users']:,}/{stats['users']:,} ({stats['tied_user_pct']:.2f}%), "
        f"tied_tail_interactions={stats['tied_interactions']:,}, max_tie={stats['max_tie']:,}"
    )
    if stats["tied_users"]:
        print("  Tie-break note: equal final time_ms values are ordered by item identity and source file.")


def _kuairand_item_count_cache_fingerprint(stage_fingerprint: str) -> str:
    payload = json.dumps(
        {
            "stage_fingerprint": stage_fingerprint,
            "cache_version": KUAIRAND_ITEM_COUNT_CACHE_VERSION,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _ensure_kuairand_item_counts(connection, stage_fingerprint: str) -> bool:
    """Build or reuse item frequencies derived from active DuckDB staging."""
    cache_fingerprint = _kuairand_item_count_cache_fingerprint(stage_fingerprint)
    metadata = _kuairand_metadata(connection)
    if (
        metadata.get("item_count_cache_fingerprint") == cache_fingerprint
        and metadata.get("item_count_cache_state") == "ready"
        and _kuairand_has_table(connection, "staged_item_counts")
    ):
        count = connection.execute("SELECT COUNT(*) FROM staged_item_counts").fetchone()[0]
        print(f"  reusing staged item-count cache: {count:,} item(s)")
        return False

    print("  building staged item-count cache (one-time stage aggregation) ...")
    _kuairand_set_metadata(connection, {"item_count_cache_state": "building"})
    connection.execute("DROP TABLE IF EXISTS staged_item_counts")
    connection.execute(
        """
        CREATE TABLE staged_item_counts AS
        SELECT item_id, COUNT(*) AS interaction_count
        FROM staged
        GROUP BY item_id
        """
    )
    _kuairand_set_metadata(
        connection,
        {
            "item_count_cache_fingerprint": cache_fingerprint,
            "item_count_cache_state": "ready",
        },
    )
    count = connection.execute("SELECT COUNT(*) FROM staged_item_counts").fetchone()[0]
    print(f"  staged item-count cache complete: {count:,} item(s)")
    if KUAIRAND_ITEM_COUNT_CACHE_CALLBACK is not None:
        _kuairand_checkpoint_database(connection)
        KUAIRAND_ITEM_COUNT_CACHE_CALLBACK("staged_item_counts")
    return True


def _kuairand_raw_source_counts(
    connection,
    processed_dir: str,
    log_paths: list[str],
    raw_parquet_fingerprint: str,
) -> dict[str, int]:
    metadata = _kuairand_metadata(connection)
    if metadata.get("source_counts_fingerprint") == raw_parquet_fingerprint and "source_counts" in metadata:
        return json.loads(metadata["source_counts"])

    expected_names = [os.path.basename(path) for path in log_paths]
    file_source_counts = json.loads(metadata.get("parquet_file_source_counts", "{}"))
    if all(name in file_source_counts for name in expected_names):
        counts = {
            "hate_interactions": sum(int(file_source_counts[name]["hate_interactions"]) for name in expected_names),
            "rand_interactions": sum(int(file_source_counts[name]["rand_interactions"]) for name in expected_names),
        }
        _kuairand_set_metadata(
            connection,
            {
                "source_counts_fingerprint": raw_parquet_fingerprint,
                "source_counts": counts,
            },
        )
        return counts

    hate_interactions, rand_interactions = connection.execute(
        f"""
        SELECT
            SUM(CASE WHEN is_hate = 1 THEN 1 ELSE 0 END),
            SUM(CASE WHEN is_rand = 1 THEN 1 ELSE 0 END)
        FROM read_parquet({_kuairand_parquet_list(processed_dir, log_paths)})
        """
    ).fetchone()
    counts = {
        "hate_interactions": int(hate_interactions or 0),
        "rand_interactions": int(rand_interactions or 0),
    }
    _kuairand_set_metadata(
        connection,
        {
            "source_counts_fingerprint": raw_parquet_fingerprint,
            "source_counts": counts,
        },
    )
    return counts


# KuaiRand recipe diagnostics and provenance
#
# The frequency CSV and analysis.json are intentionally kept lightweight. They
# explain how floors were selected (for example, the 27K item_floor=555 boundary)
# and preserve enough aggregate evidence for reproducibility notes without
# writing a training dataset during exploratory runs.
def _kuairand_item_frequency_profile(connection) -> list[dict[str, int]]:
    """Return cumulative interaction-stage item counts at every observed floor."""
    profile = []
    cumulative_items = 0
    cumulative_interactions = 0
    for floor, exact_items in connection.execute(
        """
        SELECT interaction_count, COUNT(*)
        FROM staged_item_counts
        GROUP BY interaction_count
        ORDER BY interaction_count DESC
        """
    ).fetchall():
        cumulative_items += int(exact_items)
        cumulative_interactions += int(floor) * int(exact_items)
        profile.append(
            {
                "floor": int(floor),
                "items_at_or_above": cumulative_items,
                "interactions_at_or_above": cumulative_interactions,
            }
        )
    return profile


def _write_kuairand_frequency_profile(profile: list[dict[str, int]], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    partial = f"{path}.partial"
    with open(partial, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=("floor", "items_at_or_above", "interactions_at_or_above"),
        )
        writer.writeheader()
        writer.writerows(profile)
    os.replace(partial, path)
    print(f"  wrote interaction-stage item frequency profile to {path}")


def _kuairand_target_profile_brackets(
    profile: list[dict[str, int]], target_items: int
) -> dict[str, dict[str, int] | None]:
    fewer_or_equal = [row for row in profile if row["items_at_or_above"] <= target_items]
    more_or_equal = [row for row in profile if row["items_at_or_above"] >= target_items]
    return {
        "at_or_below_target_items": max(fewer_or_equal, key=lambda row: row["items_at_or_above"], default=None),
        "at_or_above_target_items": min(more_or_equal, key=lambda row: row["items_at_or_above"], default=None),
    }


def _print_kuairand_target_profile_brackets(brackets: dict[str, dict[str, int] | None]):
    """Print the item-floor boundaries around the 7,550-item reference scale.

    7,550 is both the confirmed Pure output catalog size and the FuXi-alpha
    Table 1 item count. The interaction deltas below are diagnostic only; the
    maintained 27K recipe is justified by its engagement rule and catalog scale,
    not by checksum matching this table.
    """
    print("  Interaction-stage frequency guidance (before user filtering):")
    expected_interactions = KUAIRAND_FUXI_ALPHA_STATS["interactions"]
    labels = {
        "at_or_below_target_items": "at/below 7,550 items",
        "at_or_above_target_items": "at/above 7,550 items",
    }
    printed = set()
    for key, label in labels.items():
        row = brackets[key]
        if row is None:
            print(f"    {label}: no observed frequency boundary")
            continue
        identity = (row["floor"], row["items_at_or_above"], row["interactions_at_or_above"])
        if identity in printed:
            continue
        printed.add(identity)
        interaction_delta = row["interactions_at_or_above"] - expected_interactions
        print(
            f"    {label}: floor={row['floor']:,}, items={row['items_at_or_above']:,}, "
            f"interactions={row['interactions_at_or_above']:,} "
            f"(interaction delta {interaction_delta:+,})"
        )


def _build_kuairand_filtered_with_item_floor(connection, item_floor: int):
    """Materialize only staged rows whose items pass the first item-floor step."""
    connection.execute("DROP TABLE IF EXISTS filtered")
    connection.execute("DROP TABLE IF EXISTS eligible_items")
    print("  finding items that meet the initial item floor ...")
    connection.execute(
        """
        CREATE TEMP TABLE eligible_items AS
        SELECT item_id
        FROM staged_item_counts
        WHERE interaction_count >= ?
        """,
        (item_floor,),
    )
    eligible_count = connection.execute("SELECT COUNT(*) FROM eligible_items").fetchone()[0]
    print(f"  initial item floor retained {eligible_count:,} item(s); materializing their interactions ...")
    connection.execute(
        """
        CREATE TABLE filtered AS
        SELECT s.*
        FROM staged AS s
        JOIN eligible_items AS e USING (item_id)
        """
    )
    connection.execute("DROP TABLE eligible_items")


def _delete_below_floor(connection, column: str, floor: int) -> int:
    before = int(connection.execute("SELECT COUNT(*) FROM filtered").fetchone()[0])
    connection.execute(
        f"""
        DELETE FROM filtered
        WHERE {column} IN (
            SELECT {column}
            FROM filtered
            GROUP BY {column}
            HAVING COUNT(*) < ?
        )
        """,
        (floor,),
    )
    after = int(connection.execute("SELECT COUNT(*) FROM filtered").fetchone()[0])
    return before - after


def _filter_kuairand_staging(
    connection,
    item_floor: int = KUAIRAND_ITEM_FLOOR,
    user_floor: int = KUAIRAND_USER_FLOOR,
    exclude_hate: bool = False,
    interaction_rule: str = KUAIRAND_DEFAULT_INTERACTION_RULE,
    exclude_rand: bool = False,
) -> list[dict[str, float | int]]:
    """Apply alternating item/user count floors until the retained core converges.

    Repeated impressions are preserved. We start with an item prefilter because
    it is the high-leverage cut for 27K, then alternate user and item floors to a
    fixed point. The reported "removed" value is interactions removed, not users
    or items.
    """
    iteration_stats = []
    print(
        f"  active recipe: interaction_rule={_kuairand_interaction_rule_version(interaction_rule)}, "
        f"exclude_hate={exclude_hate}, exclude_rand={exclude_rand}, "
        f"item_floor={item_floor}, user_floor={user_floor}"
    )
    _build_kuairand_filtered_with_item_floor(connection, item_floor)
    _print_kuairand_stats("after initial item floor", _kuairand_stats(connection, "filtered"))

    removed = _delete_below_floor(connection, "user_id", user_floor)
    stats = _kuairand_stats(connection, "filtered")
    iteration_stats.append({"iteration": 1, "removed": removed, **stats})
    _print_kuairand_stats(f"filter iteration 1 (removed {removed:,} interactions)", stats)
    if removed == 0:
        return iteration_stats

    iteration = 1
    while True:
        iteration += 1
        removed_items = _delete_below_floor(connection, "item_id", item_floor)
        removed_users = _delete_below_floor(connection, "user_id", user_floor)
        removed = removed_items + removed_users
        stats = _kuairand_stats(connection, "filtered")
        iteration_stats.append({"iteration": iteration, "removed": removed, **stats})
        _print_kuairand_stats(f"filter iteration {iteration} (removed {removed:,} interactions)", stats)
        if removed == 0:
            return iteration_stats


def _kuairand_value_distribution(connection, table: str, column: str) -> list[dict[str, int]]:
    return [
        {"value": int(value), "interactions": int(count)}
        for value, count in connection.execute(
            f"SELECT {column}, COUNT(*) FROM {table} GROUP BY {column} ORDER BY {column}"
        ).fetchall()
    ]


def _kuairand_behavior_breakdown(connection, table: str) -> dict[str, int]:
    row = connection.execute(
        f"""
        SELECT
            SUM(CASE WHEN is_hate = 1 THEN 1 ELSE 0 END),
            SUM(CASE WHEN long_view = 1 THEN 1 ELSE 0 END),
            SUM(CASE WHEN is_click = 1 THEN 1 ELSE 0 END),
            SUM(CASE WHEN is_like = 1 THEN 1 ELSE 0 END),
            SUM(CASE WHEN is_follow = 1 THEN 1 ELSE 0 END),
            SUM(CASE WHEN is_comment = 1 THEN 1 ELSE 0 END),
            SUM(CASE WHEN is_forward = 1 THEN 1 ELSE 0 END),
            SUM(CASE WHEN is_rand = 1 THEN 1 ELSE 0 END)
        FROM {table}
        """
    ).fetchone()
    keys = ("hate", "long_view", "click", "like", "follow", "comment", "forward", "rand")
    return {key: int(value or 0) for key, value in zip(keys, row)}


def _report_kuairand_alpha_diagnostic(
    variant: str, stats: dict[str, float | int], outputs_will_be_written: bool = True
):
    if variant != "27k":
        print("  Smoke-test variant only; these statistics do not validate the FuXi-alpha benchmark.")
        return

    expected = KUAIRAND_FUXI_ALPHA_STATS
    compatible = (
        stats["users"] == expected["users"]
        and stats["items"] == expected["items"]
        and stats["interactions"] == expected["interactions"]
        and abs(stats["avg_length"] - expected["avg_length"]) < 0.005
    )
    if compatible:
        print("  BENCHMARK COMPATIBLE: all FuXi-alpha Table 1 checksums match.")
    else:
        print("\n  FuXi-alpha Table 1 diagnostic: this recipe does not match the published checksum.")
        print(
            f"  Expected users={expected['users']:,}, items={expected['items']:,}, "
            f"interactions={expected['interactions']:,}, avg_length={expected['avg_length']:.2f}."
        )
        for key in ("users", "items", "interactions"):
            delta = stats[key] - expected[key]
            print(f"  Delta {key}: {delta:+,}")
        avg_delta = stats["avg_length"] - expected["avg_length"]
        print(f"  Delta avg_length: {avg_delta:+.2f}")
        if stats["items"] > expected["items"]:
            print("  Hint: item count is high; raise --kuairand-item-floor.")
        elif stats["items"] < expected["items"]:
            print("  Hint: item count is low; lower --kuairand-item-floor.")
        if stats["users"] > expected["users"]:
            print("  Hint: user count is high; raise --kuairand-user-floor.")
        elif stats["users"] < expected["users"]:
            print("  Hint: user count is low; lower --kuairand-user-floor.")
        if outputs_will_be_written:
            print("  Outputs will be written as this recipe's dataset, not as a benchmark reproduction.\n")
        else:
            print("  Analysis-only run: training outputs will not be written.\n")


def _kuairand_target_deltas(stats: dict[str, float | int]) -> dict[str, float | int]:
    expected = KUAIRAND_FUXI_ALPHA_STATS
    return {
        "users": stats["users"] - expected["users"],
        "items": stats["items"] - expected["items"],
        "interactions": stats["interactions"] - expected["interactions"],
        "avg_length": stats["avg_length"] - expected["avg_length"],
    }


def _write_kuairand_analysis(path: str, analysis: dict):
    """Write preprocessing diagnostics without training outputs."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    partial = f"{path}.partial"
    with open(partial, "w") as f:
        json.dump(analysis, f, indent=2)
    os.replace(partial, path)
    print(f"  wrote KuaiRand recipe diagnostics to {path}")


def _print_kuairand_disk_guidance(variant: str, archive_path: str):
    if variant != "27k":
        return
    print("  27K disk note: budget roughly 90-100GB during first DuckDB/Parquet preprocessing.")
    print("  This includes the archive, extracted standard logs, typed Parquet, and DuckDB stage/filter tables.")
    print(f"  Use --kuairand-delete-archive-after-extract to remove {archive_path} after extraction.")


def _validate_kuairand_recipe(
    item_floor: int,
    user_floor: int,
    interaction_rule: str = KUAIRAND_DEFAULT_INTERACTION_RULE,
):
    if item_floor < 1:
        raise ValueError("--kuairand-item-floor must be >= 1")
    if user_floor < 1:
        raise ValueError("--kuairand-user-floor must be >= 1")
    if interaction_rule not in KUAIRAND_INTERACTION_RULES:
        raise ValueError(f"--kuairand-interaction-rule must be one of: {', '.join(KUAIRAND_INTERACTION_RULES)}")


def load_kuairand_raw(
    data_dir: str,
    variant: str,
    item_floor: int = KUAIRAND_ITEM_FLOOR,
    user_floor: int = KUAIRAND_USER_FLOOR,
    delete_archive_after_extract: bool = False,
    analysis_only: bool = False,
    exclude_hate: bool = False,
    interaction_rule: str = KUAIRAND_DEFAULT_INTERACTION_RULE,
    exclude_rand: bool = False,
):
    """Acquire, stage, filter, and load a KuaiRand interaction slice.

    KuaiRand now uses a DuckDB analytical pipeline backed by typed Parquet.
    Repeated impressions are intentionally preserved; dedup support was removed
    because the current benchmark recipes do not use it.
    """
    started_at = time.monotonic()
    _validate_kuairand_recipe(item_floor, user_floor, interaction_rule)
    archive_name, _ = KUAIRAND_VARIANTS[variant]
    archive_path = os.path.join(data_dir, archive_name)
    processed_dir = _kuairand_processed_dir(data_dir, variant)
    database_path = _kuairand_duckdb_path(processed_dir)
    first_duckdb = not os.path.exists(database_path)
    if first_duckdb:
        _print_kuairand_disk_guidance(variant, archive_path)
    log_paths = download_kuairand(variant, data_dir, delete_archive_after_extract)
    os.makedirs(processed_dir, exist_ok=True)

    connection = _kuairand_connect(database_path)
    try:
        raw_parquet_fingerprint = _ensure_kuairand_raw_parquet(connection, log_paths, variant, processed_dir)
        stage_fingerprint, stage_reused = _ensure_kuairand_stage(
            connection,
            processed_dir,
            log_paths,
            raw_parquet_fingerprint,
            interaction_rule,
            exclude_hate,
            exclude_rand,
        )
        filter_fingerprint = _kuairand_filter_fingerprint(stage_fingerprint, item_floor, user_floor)

        _ensure_kuairand_item_counts(connection, stage_fingerprint)
        metadata = _kuairand_metadata(connection)
        source_counts = _kuairand_raw_source_counts(connection, processed_dir, log_paths, raw_parquet_fingerprint)
        print(f"  source hate rows: {source_counts['hate_interactions']:,}")
        print(f"  source random rows: {source_counts['rand_interactions']:,}")
        stage_stats = _kuairand_stats(connection, "staged")
        _print_kuairand_stats("interaction stage", stage_stats)
        profile = _kuairand_item_frequency_profile(connection)
        profile_path = _kuairand_profile_path(processed_dir, exclude_hate, interaction_rule, exclude_rand)
        _write_kuairand_frequency_profile(profile, profile_path)
        profile_brackets = _kuairand_target_profile_brackets(profile, KUAIRAND_FUXI_ALPHA_STATS["items"])
        if variant == "27k":
            _print_kuairand_target_profile_brackets(profile_brackets)

        if (
            metadata.get("filter_fingerprint") != filter_fingerprint
            or metadata.get("filter_state") != "ready"
            or not _kuairand_has_table(connection, "filtered")
        ):
            filter_reused = False
            _kuairand_set_metadata(connection, {"filter_state": "building"})
            iteration_stats = _filter_kuairand_staging(
                connection,
                item_floor,
                user_floor,
                exclude_hate,
                interaction_rule,
                exclude_rand,
            )
            _kuairand_set_metadata(
                connection,
                {
                    "filter_fingerprint": filter_fingerprint,
                    "filter_state": "ready",
                    "filter_iterations": iteration_stats,
                },
            )
            if KUAIRAND_ITEM_COUNT_CACHE_CALLBACK is not None:
                _kuairand_checkpoint_database(connection)
                KUAIRAND_ITEM_COUNT_CACHE_CALLBACK("filtered recipe")
        else:
            filter_reused = True
            iteration_stats = json.loads(metadata.get("filter_iterations", "[]"))
            print(
                f"  reusing filtered recipe: interaction_rule={_kuairand_interaction_rule_version(interaction_rule)}, "
                f"exclude_hate={exclude_hate}, exclude_rand={exclude_rand}, "
                f"item_floor={item_floor}, user_floor={user_floor}"
            )
            if not iteration_stats:
                print("  prior filter iteration history is unavailable; new recipe builds will record it")

        final_stats = _kuairand_stats(connection, "filtered")
        _print_kuairand_stats("final", final_stats)
        tie_stats = _kuairand_final_timestamp_tie_stats(connection)
        _print_kuairand_tie_stats(tie_stats)
        _report_kuairand_alpha_diagnostic(variant, final_stats, outputs_will_be_written=not analysis_only)

        stage_breakdown = _kuairand_behavior_breakdown(connection, "staged")
        filtered_breakdown = _kuairand_behavior_breakdown(connection, "filtered")
        stage_tab_distribution = _kuairand_value_distribution(connection, "staged", "tab")
        filtered_tab_distribution = _kuairand_value_distribution(connection, "filtered", "tab")
        stage_rand_distribution = _kuairand_value_distribution(connection, "staged", "is_rand")
        filtered_rand_distribution = _kuairand_value_distribution(connection, "filtered", "is_rand")

        output_dataset = (
            f"kuairand-{variant}-"
            f"{kuairand_recipe_slug(item_floor, user_floor, exclude_hate, interaction_rule, exclude_rand)}"
        )
        global KUAIRAND_LAST_RUN_METADATA
        KUAIRAND_LAST_RUN_METADATA = {
            "raw_parquet_fingerprint": raw_parquet_fingerprint,
            "stage_fingerprint": stage_fingerprint,
            "filter_fingerprint": filter_fingerprint,
            "stage_stats": stage_stats,
            "final_stats": final_stats,
            "final_timestamp_ties": tie_stats,
            "source_hate_interactions": source_counts["hate_interactions"],
            "source_rand_interactions": source_counts["rand_interactions"],
            "stage_tab_distribution": stage_tab_distribution,
            "filtered_tab_distribution": filtered_tab_distribution,
            "stage_rand_distribution": stage_rand_distribution,
            "filtered_rand_distribution": filtered_rand_distribution,
            "stage_behavior_breakdown": stage_breakdown,
            "filtered_behavior_breakdown": filtered_breakdown,
        }
        if analysis_only:
            analysis_path = os.path.join(data_dir, output_dataset, "analysis.json")
            analysis = {
                "dataset": f"kuairand-{variant}",
                "output_dataset": output_dataset,
                "analysis_only": True,
                "recipe": {
                    "item_floor": item_floor,
                    "user_floor": user_floor,
                    "interaction_rule": _kuairand_interaction_rule_version(interaction_rule),
                    "exclude_hate": exclude_hate,
                    "exclude_rand": exclude_rand,
                    "filter_implementation": KUAIRAND_FILTER_IMPLEMENTATION_VERSION,
                    "tab_scope_note": "tab=1 is empirical; KuaiRand does not publish tab value semantics.",
                },
                "raw_parquet_fingerprint": raw_parquet_fingerprint,
                "stage_fingerprint": stage_fingerprint,
                "filter_fingerprint": filter_fingerprint,
                "stage_reused": stage_reused,
                "filter_reused": filter_reused,
                "stage_stats": stage_stats,
                "source_hate_interactions": source_counts["hate_interactions"],
                "source_rand_interactions": source_counts["rand_interactions"],
                "frequency_profile": {
                    "scope": "interaction stage before user filtering",
                    "path": os.path.relpath(profile_path, data_dir),
                    "target_items": KUAIRAND_FUXI_ALPHA_STATS["items"],
                    "target_brackets": profile_brackets,
                },
                "filter_iterations_available": bool(iteration_stats),
                "filter_iterations": iteration_stats,
                "final_stats": final_stats,
                "final_timestamp_ties": tie_stats,
                "stage_tab_distribution": stage_tab_distribution,
                "filtered_tab_distribution": filtered_tab_distribution,
                "stage_rand_distribution": stage_rand_distribution,
                "filtered_rand_distribution": filtered_rand_distribution,
                "stage_behavior_breakdown": stage_breakdown,
                "filtered_behavior_breakdown": filtered_breakdown,
                "fuxi_alpha_target": KUAIRAND_FUXI_ALPHA_STATS,
                "fuxi_alpha_deltas": _kuairand_target_deltas(final_stats) if variant == "27k" else None,
                "elapsed_seconds": time.monotonic() - started_at,
            }
            _write_kuairand_analysis(analysis_path, analysis)
            return None

        interactions = []
        previous_user = None
        previous_timestamp = None
        # Ordering uses the event timestamp first. Stable item identity and the
        # source file provide deterministic tie-breaks for equal timestamps.
        for user, item, timestamp in connection.execute(
            "SELECT user_id, item_id, time_ms FROM filtered ORDER BY user_id, time_ms, item_id, source_file"
        ).fetchall():
            if user == previous_user and timestamp < previous_timestamp:
                raise ValueError(f"Non-monotonic KuaiRand timestamps for user {user}")
            previous_user = user
            previous_timestamp = timestamp
            interactions.append((str(user), str(item), int(timestamp)))
        return interactions
    finally:
        connection.close()


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
        # NOTE: timestamps are ordering values. They do not need reindexing like
        # user/item IDs, and may be seconds or milliseconds depending on source.
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
        # Expected stats after this repo's preprocessing:
        #   Users: 6,040  Items: 3,416  Avg actions/user: 165.5
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
        # Expected stats after this repo's preprocessing:
        #   Users: 138,493  Items: 18,345  Avg actions/user: 144.3
    },
    "kuairand-27k": {
        "loader": lambda data_dir, args: load_kuairand_raw(
            data_dir,
            variant="27k",
            item_floor=args.kuairand_item_floor,
            user_floor=args.kuairand_user_floor,
            delete_archive_after_extract=args.kuairand_delete_archive_after_extract,
            analysis_only=args.kuairand_analysis_only,
            exclude_hate=args.kuairand_exclude_hate,
            interaction_rule=args.kuairand_interaction_rule,
            exclude_rand=args.kuairand_exclude_rand,
        ),
        "kcore": False,  # custom DuckDB-backed frequency filtering in the loader
        "titles_writer": None,
        # Maintained 27K recipe:
        #   --kuairand-interaction-rule long-view-not-hate-tab1
        #   --kuairand-exclude-rand --kuairand-item-floor 555 --kuairand-user-floor 5
        # This is a principled long-view engagement slice, not an alpha checksum target.
    },
    "kuairand-pure": {
        "loader": lambda data_dir, args: load_kuairand_raw(
            data_dir,
            variant="pure",
            item_floor=args.kuairand_item_floor,
            user_floor=args.kuairand_user_floor,
            delete_archive_after_extract=args.kuairand_delete_archive_after_extract,
            analysis_only=args.kuairand_analysis_only,
            exclude_hate=args.kuairand_exclude_hate,
            interaction_rule=args.kuairand_interaction_rule,
            exclude_rand=args.kuairand_exclude_rand,
        ),
        "kcore": False,  # smoke test of the same custom pipeline, not the benchmark
        "titles_writer": None,
    },
}


def main():
    parser = argparse.ArgumentParser(description="nanoSASRec data preprocessing")
    parser.add_argument(
        "--dataset", type=str, default="ml-1m", choices=list(DATASETS.keys()), help="Dataset to preprocess"
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory")
    parser.add_argument(
        "--kuairand-item-floor", type=int, default=KUAIRAND_ITEM_FLOOR, help="KuaiRand item frequency floor"
    )
    parser.add_argument(
        "--kuairand-user-floor", type=int, default=KUAIRAND_USER_FLOOR, help="KuaiRand user activity floor"
    )
    parser.add_argument(
        "--kuairand-interaction-rule",
        type=str,
        default=KUAIRAND_DEFAULT_INTERACTION_RULE,
        choices=KUAIRAND_INTERACTION_RULES,
        help="KuaiRand interaction materialization rule",
    )
    parser.add_argument(
        "--kuairand-exclude-hate",
        action="store_true",
        help="Exclude rows where is_hate == 1; default keeps all impressions",
    )
    parser.add_argument(
        "--kuairand-exclude-rand",
        action="store_true",
        help="Exclude rows where is_rand == 1; default keeps all standard-log rows",
    )
    parser.add_argument(
        "--kuairand-delete-archive-after-extract",
        action="store_true",
        help="Delete KuaiRand .tar.gz archive after the required standard logs have been extracted",
    )
    parser.add_argument(
        "--kuairand-flat-output",
        action="store_true",
        help="Write KuaiRand outputs to data/<dataset> instead of a recipe-stamped directory",
    )
    parser.add_argument(
        "--kuairand-analysis-only",
        action="store_true",
        help="Report and persist KuaiRand recipe diagnostics without writing training outputs",
    )
    args = parser.parse_args()

    ds = DATASETS[args.dataset]
    print(f"\n{'=' * 60}")
    print(f"  Preprocessing: {args.dataset}")
    print(f"{'=' * 60}")

    # step 1: load raw interactions
    print("\n[1/4] Loading raw data ...")
    os.makedirs(args.data_dir, exist_ok=True)
    if args.dataset.startswith("kuairand-"):
        raw = ds["loader"](args.data_dir, args)
    else:
        raw = ds["loader"](args.data_dir)
    if raw is None:
        print("\n  Analysis-only run complete; skipped interaction export, splitting, and training outputs.")
        print(f"\n{'=' * 60}")
        print("  Done!")
        print(f"{'=' * 60}\n")
        return
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
    output_dataset = kuairand_output_dataset_name(args.dataset, args)
    out_dir = os.path.join(args.data_dir, output_dataset)
    txt_path = os.path.join(out_dir, output_txt_filename(output_dataset))
    print("\n[4/4] Writing outputs ...")
    if output_dataset != args.dataset:
        print(f"  KuaiRand recipe output directory: {out_dir}")
    write_txt(dataset, txt_path)
    write_split(dataset, out_dir)
    write_kuairand_recipe_metadata(out_dir, args.dataset, args)

    # write item titles if available (for human-readable predictions)
    titles_writer = ds.get("titles_writer")
    if titles_writer:
        titles_writer(args.data_dir, out_dir)

    print(f"\n{'=' * 60}")
    print("  Done! Run train v1 through v10 to start training.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
