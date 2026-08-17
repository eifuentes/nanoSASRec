"""
remote.py - Modal wrappers for nanoSASRec preprocessing and training.

KuaiRand preprocessing is CPU/IO-heavy and should run separately from GPU
training. Each base dataset gets its own Modal Volume mounted at
/volumes/<dataset>; recipe-stamped KuaiRand outputs live inside that volume
while the expensive raw Parquet and DuckDB stage/cache artifacts are reused
across sweeps.

Usage:
    # CPU-only Pure/FuXi-gamma slice prep
    uv run modal run remote.py::prep --cmd "python -u data.py --dataset kuairand-pure --kuairand-item-floor 1 --kuairand-user-floor 5"

    # CPU-only 27K long-sequence prep; detached so laptop sleep does not stop the job
    uv run modal run --detach remote.py::prep_job --cmd "python -u data.py --dataset kuairand-27k --kuairand-interaction-rule long-view-not-hate-tab1 --kuairand-exclude-rand --kuairand-item-floor 555 --kuairand-user-floor 5 --kuairand-delete-archive-after-extract"

    # CPU-only 27K sweep; reuses raw Parquet and rebuilds only active DuckDB stage/filter output
    uv run modal run --detach remote.py::prep_job --cmd "python -u data.py --dataset kuairand-27k --kuairand-interaction-rule click-not-hate-tab1 --kuairand-exclude-rand --kuairand-item-floor 5 --kuairand-user-floor 5 --kuairand-analysis-only"

    # CPU-only recipe diagnostics; skips Python interaction export and training files
    uv run modal run --detach remote.py::prep_job --cmd "python -u data.py --dataset kuairand-27k --kuairand-interaction-rule long-view-not-hate-tab1 --kuairand-exclude-rand --kuairand-item-floor 555 --kuairand-user-floor 5 --kuairand-analysis-only"

    # GPU training against a prepared recipe-stamped dataset
    uv run modal run --detach remote.py::train --cmd "python -u train_v9.py --dataset kuairand-27k-rule-long-view-not-hate-tab1-item555-user5-exclude-rand --epochs 200 --eval-every 20 --device cuda"

    # GPU convenience entrypoint; appends --device cuda, but assumes data is already prepped.
    # Do not pass data.py commands here; use remote.py::prep for preprocessing.
    uv run modal run --detach remote.py --cmd "python -u train_v9.py --dataset ml-20m"
"""

import glob
import os
import shlex
import subprocess
import sys

try:
    import modal
except ModuleNotFoundError:  # Allows local helper tests without the cloud dependency group.
    modal = None


BASE_DATASETS = ("ml-1m", "ml-20m", "kuairand-pure", "kuairand-27k")
APP_DIR = "/app"
VOLUME_ROOT = "/volumes"
VOLUME_NAMES = {dataset: f"nanoSASRec-{dataset}" for dataset in BASE_DATASETS}
# Modal validates ephemeral disks as coarse tiers; 512 GiB is the current lower bound.
EPHEMERAL_DISK_MIB = 512 * 1024


def base_dataset_for(dataset: str) -> str:
    """Map exact or recipe-stamped dataset names to their base dataset."""
    if dataset in BASE_DATASETS:
        return dataset
    for base in BASE_DATASETS:
        if dataset.startswith(f"{base}-"):
            return base
    raise ValueError(
        f"Cannot infer base dataset from {dataset!r}; expected one of {', '.join(BASE_DATASETS)} "
        "or a recipe-stamped name with one of those prefixes"
    )


def volume_data_dir(base_dataset: str) -> str:
    """Return the mounted data directory for a base dataset volume."""
    return f"{VOLUME_ROOT}/{base_dataset}"


def _option_value(args: list[str], option: str) -> str | None:
    for index, token in enumerate(args):
        if token == option:
            return args[index + 1] if index + 1 < len(args) else None
        if token.startswith(f"{option}="):
            return token.split("=", 1)[1]
    return None


def normalize_cmd_data_dir(cmd: str) -> str:
    """Force the command to use the inferred dataset volume."""
    args = shlex.split(cmd)
    dataset = _option_value(args, "--dataset")
    if dataset is None:
        raise ValueError("Remote commands must include --dataset so their Modal volume can be selected.")

    normalized_args = []
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--data-dir":
            if index + 1 >= len(args):
                raise ValueError("--data-dir requires a value; remote commands select this path automatically.")
            index += 2
            continue
        if token.startswith("--data-dir="):
            index += 1
            continue
        normalized_args.append(token)
        index += 1

    normalized_args.extend(["--data-dir", volume_data_dir(base_dataset_for(dataset))])
    return shlex.join(normalized_args)


def _cmd_dataset(cmd: str) -> str | None:
    return _option_value(shlex.split(cmd), "--dataset")


def _is_python_data_command(args: list[str]) -> bool:
    return any(token == "data.py" or token.endswith("/data.py") for token in args)


def _require_data_stage_callback(data_module):
    if not hasattr(data_module, "KUAIRAND_STAGE_FILE_CALLBACK"):
        raise RuntimeError(
            "data.py is missing KUAIRAND_STAGE_FILE_CALLBACK, which remote.py uses to checkpoint Modal volumes "
            "after each KuaiRand Parquet conversion or stage milestone."
        )
    if not hasattr(data_module, "KUAIRAND_DOWNLOAD_CHECKPOINT_CALLBACK"):
        raise RuntimeError(
            "data.py is missing KUAIRAND_DOWNLOAD_CHECKPOINT_CALLBACK, which remote.py uses to checkpoint Modal "
            "volumes during large KuaiRand archive downloads."
        )
    if not hasattr(data_module, "KUAIRAND_ITEM_COUNT_CACHE_CALLBACK"):
        raise RuntimeError(
            "data.py is missing KUAIRAND_ITEM_COUNT_CACHE_CALLBACK, which remote.py uses to checkpoint the reusable "
            "KuaiRand item-count cache."
        )


def _reject_data_command_for_gpu_entrypoint(cmd: str):
    if _is_python_data_command(shlex.split(cmd)):
        raise ValueError("remote.py's default GPU entrypoint does not run data.py; use remote.py::prep instead.")


def _build_image():
    # Keep one image for prep and training so local scripts and Modal runs see
    # the same dependency set. CPU-only prep therefore carries torch too, but
    # Modal caches the image and the simpler mental model is worth it here.
    image = modal.Image.debian_slim(python_version="3.12").uv_sync().add_local_python_source("sasrec")
    for filename in sorted(glob.glob("*.py")):
        image = image.add_local_file(filename, f"{APP_DIR}/{filename}")
    return image


def _reload_volumes(volumes):
    for volume in volumes.values():
        volume.reload()


def _commit_volumes(volumes):
    for volume in volumes.values():
        volume.commit()


def _selected_volumes(cmd: str, volumes):
    dataset = _cmd_dataset(cmd)
    if dataset is None:
        return volumes
    base = base_dataset_for(dataset)
    return {base: volumes[base]}


def _run_data_py_in_process(cmd: str, volumes):
    """Run data.py in-process so KuaiRand file-stage checkpoints can commit Modal volumes."""
    args = shlex.split(cmd)
    data_py_index = next(index for index, token in enumerate(args) if token == "data.py" or token.endswith("/data.py"))
    data_args = args[data_py_index + 1 :]

    if APP_DIR not in sys.path:
        sys.path.insert(0, APP_DIR)
    import data

    _require_data_stage_callback(data)
    selected_volumes = _selected_volumes(cmd, volumes)

    def commit_after_stage_file(source_name: str):
        print(f"  Modal checkpoint after KuaiRand preprocessing milestone {source_name}: committing volume(s).")
        _commit_volumes(selected_volumes)

    def commit_after_download_checkpoint(label: str):
        print(f"  Modal checkpoint after downloading {label}: committing volume(s).")
        _commit_volumes(selected_volumes)

    def commit_after_item_count_cache(label: str):
        print(f"  Modal checkpoint after building {label}: committing volume(s).")
        _commit_volumes(selected_volumes)

    old_argv = sys.argv
    old_stage_callback = data.KUAIRAND_STAGE_FILE_CALLBACK
    old_download_callback = data.KUAIRAND_DOWNLOAD_CHECKPOINT_CALLBACK
    old_item_count_cache_callback = data.KUAIRAND_ITEM_COUNT_CACHE_CALLBACK
    old_cwd = os.getcwd()
    try:
        sys.argv = ["data.py", *data_args]
        data.KUAIRAND_STAGE_FILE_CALLBACK = commit_after_stage_file
        data.KUAIRAND_DOWNLOAD_CHECKPOINT_CALLBACK = commit_after_download_checkpoint
        data.KUAIRAND_ITEM_COUNT_CACHE_CALLBACK = commit_after_item_count_cache
        os.chdir(APP_DIR)
        try:
            data.main()
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise
    finally:
        os.chdir(old_cwd)
        data.KUAIRAND_STAGE_FILE_CALLBACK = old_stage_callback
        data.KUAIRAND_DOWNLOAD_CHECKPOINT_CALLBACK = old_download_callback
        data.KUAIRAND_ITEM_COUNT_CACHE_CALLBACK = old_item_count_cache_callback
        sys.argv = old_argv


if modal is not None:
    app = modal.App("nanoSASRec")
    image = _build_image()
    dataset_volumes = {
        dataset: modal.Volume.from_name(volume_name, create_if_missing=True)
        for dataset, volume_name in VOLUME_NAMES.items()
    }
    volume_mounts = {volume_data_dir(dataset): volume for dataset, volume in dataset_volumes.items()}

    @app.function(
        image=image,
        timeout=86400,  # 24 hours
        volumes=volume_mounts,
        ephemeral_disk=EPHEMERAL_DISK_MIB,
    )
    def prep(cmd: str):
        normalized_cmd = normalize_cmd_data_dir(cmd)
        print(f"Running prep: {normalized_cmd}")
        selected_volumes = _selected_volumes(normalized_cmd, dataset_volumes)
        _reload_volumes(selected_volumes)
        if _is_python_data_command(shlex.split(normalized_cmd)):
            _run_data_py_in_process(normalized_cmd, selected_volumes)
        else:
            subprocess.run(normalized_cmd, shell=True, cwd=APP_DIR, check=True)
        _commit_volumes(selected_volumes)

    @app.function(
        image=image,
        gpu="A10",
        timeout=28800,  # 8 hours
        volumes=volume_mounts,
    )
    def train(cmd: str):
        normalized_cmd = normalize_cmd_data_dir(cmd)
        print(f"Running train: {normalized_cmd}")
        selected_volumes = _selected_volumes(normalized_cmd, dataset_volumes)
        _reload_volumes(selected_volumes)
        subprocess.run(normalized_cmd, shell=True, cwd=APP_DIR, check=True)
        _commit_volumes(selected_volumes)

    @app.local_entrypoint()
    def prep_job(cmd: str):
        normalized_cmd = normalize_cmd_data_dir(cmd)
        print(f"Running prep: {normalized_cmd}")
        prep.remote(normalized_cmd)

    @app.local_entrypoint()
    def main(cmd: str):
        _reject_data_command_for_gpu_entrypoint(cmd)
        cmd = f"{cmd} --device cuda"
        print(f"Running: {cmd}")
        train.spawn(cmd)
